"""Pure-NumPy reference semantics for the cross-library goldens.

This module is the neutral specification of the behaviour both libraries
(plastax and plastax-cpp) must reproduce. It deliberately imports neither:
every quantity is computed with NumPy integer/float32 arithmetic so a golden
regenerates byte-identically on any machine, and a disagreement between a
library and a golden is a defect in the library (or a deliberate spec change,
made by editing this file and regenerating).

State model
-----------
A network state is plain data:

- ``units``: a list of dicts ``{id, level, pruned, is_input, is_output,
  fields: {name: float}}``. Unit ids are dense ``[0, len(units))``; a unit is
  *live* iff it is allocated and not pruned. The list order is id order.
- ``edges``: a list of dicts ``{src, dst, fields: {name: float}}`` in slot
  order (the order the implementation stores them). Goldens compare edges as a
  multiset sorted by ``(src, dst, field values)`` so slot layout stays a
  backend choice.

Randomness
----------
``philox32(seed, counter)`` is the 10-round Philox-4x32 keyed generator both
libraries implement; its raw word stream is pinned bit-exactly by
``tests/golden/rng_philox32.json``. Rule draws use the site contract:

    site_seed = (philox32(network_seed, (step << 8) | stream) << 32) | key
    draw(j, sub) = philox32(site_seed, (j << 16) | sub)      # sub < 2**16

with ``stream = 1`` for growth, ``key`` the proposer key (unit id;
``philox32((src << 32) | dst, occurrence)`` for a connection; 0 for global),
``j`` the proposal index, and ``sub`` advancing once per draw inside one
proposal. Derived draws:

    uniform(word)       = float32(word >> 8) * 2**-24          # [0, 1)
    uniform_int(word,n) = min(uint32(uniform(word) * n), n-1)

Unit lifecycle
--------------
- Capacity is fixed; unit arrays never grow. Free ids are pruned ids below the
  allocation high-water mark plus all ids in [high-water, capacity).
- ``update_unit`` runs on every live unit, inputs and outputs included.
- Pruning is permanent. Input and output units are exempt: the predicate is
  not evaluated on them. A pruned unit's incident connections die in the same
  phase and its fields reset to their declared defaults.
- Addition evaluates the rule on every unit live at the start of the phase, in
  ascending id order; each parent spawns at most one child per step, and the
  i-th spawning parent receives the i-th lowest free id. Ids freed by this
  step's prune phase are reusable in the same step. Children do not act as
  parents in the step that created them. A spawn with no free id left is
  dropped and raises ``unit_overflow``. The child's level is
  ``clamp(parent_level + offset, 1, max_levels - 1)``; ``init_unit`` writes
  its fields, unwritten fields take their declared defaults; the child starts
  with no connections.

Loss
----
The loss is one call over every output unit: it returns the scalar loss and
writes the gradient seed dL/d(activation) of each output into its declared
seed field (``loss_grad`` here); nothing else is written. Softmax
cross-entropy uses the max-subtracted log-sum-exp form, in this float32
operation order: ``m = max(a)`` (a running ``>`` scan from ``-inf``),
``s = sum(exp(a - m))`` (output order), then per output ``z = a - m``,
``seed = exp(z) / s - t`` and ``loss += t * (log(s) - z)``.

Growth selection pipeline
-------------------------
1. Trigger: if false this step, the phase is a no-op.
2. Candidates come from live proposers only (units or connections), or from
   the exhaustive / shortlist grids. ``shortlist_per_level`` draws one grid
   per source level, levels ascending: sources are the level's top-M live
   units by importance, destinations the top-M live units inside that
   level's validity window (level gap and direction), by importance.
3. Validity (each failure scores the candidate ``-inf``): endpoints live and
   in range; ``src != dst`` unless self-loops are allowed;
   ``|level(dst) - level(src)| <= max_level_gap``; the direction constraint
   (``deeper``: strictly deeper destination; ``same_or_deeper``: at least as
   deep); in topological mode, ``level(src)`` must equal the selection
   bucket's level.
4. Non-finite scores become ``-inf``; ``-inf`` is a veto.
5. ``dedupe_live`` (off by default): veto candidates equal to a live edge.
6. ``dedupe_step`` (off by default): among equal-key candidates keep the first
   in the total order; veto the rest.
7. The total order sorts by ``(-score, src, dst, candidate_index)`` ascending,
   where the candidate index is ``unit_id * P + j`` (per-unit),
   ``r * P + j`` with r the proposing connection's rank in ascending
   ``(src, dst, occurrence)`` over live connections (per-connection), ``j``
   (global), ``src * capacity + dst`` (exhaustive), or the row-major position
   in the importance-ranked M x M grid (shortlist; importance ties break by
   ascending unit id), or ``level_rank * M * M`` plus the row-major position
   in that level's grid (shortlist_per_level; level_rank counts the levels
   holding live units, ascending). Selection runs per source level in that order --
   ``top_k`` takes the first ``max_new_per_level`` finite candidates,
   ``threshold`` those with score >= threshold(g) up to ``max_new_per_level``,
   ``all`` every finite candidate -- then ``max_new_per_step`` applies across
   levels, level ascending.
8. Selected candidates claim free connection slots per source level, each
   level's in the total order; a candidate left without a slot is dropped and
   raises ``conn_overflow``. The free slots depend on the propagation model:

   - topological: each level owns one bucket and claims only that bucket's
     free slots. A level whose bucket is full after the claim (no free slot
     left) is grown by the host loop, which then re-runs the step: the
     goldens' ``retry`` block re-runs this phase at ``step + 1`` over the
     attempt's output edges with the regrown free counts.
   - pipeline: each level first claims the dead slots whose former occupant
     was sourced at that level; the rest spill to the shared never-used tail,
     levels ascending and in the total order within a level. Overflow means
     the tail ran out. The single-domain goldens (``free_slots``) are this
     model with no dead slots.
9. ``init`` writes the new edge's fields; unwritten fields take declared
   defaults.
10. Flags: ``needs_resort`` iff a committed edge has
    ``level(dst) <= level(src)``; ``grown`` is the committed count.

Without dedupe, a candidate equal to a live edge creates a parallel edge and
two equal candidates create two edges.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

U32 = np.uint32
_MASK32 = np.uint32(0xFFFFFFFF)

# Philox-4x32-10 constants (Salmon et al., SC'11), as in both libraries.
_PHILOX_M0 = np.uint32(0xD2511F53)
_PHILOX_M1 = np.uint32(0xCD9E8D57)
_PHILOX_W0 = np.uint32(0x9E3779B9)
_PHILOX_W1 = np.uint32(0xBB67AE85)


def _mulhilo(a: np.uint32, b: np.uint32) -> tuple[np.uint32, np.uint32]:
    """Return the high and low 32-bit halves of ``a * b``."""
    prod = int(a) * int(b)
    return np.uint32((prod >> 32) & 0xFFFFFFFF), np.uint32(prod & 0xFFFFFFFF)


def philox32(seed: int, counter: int) -> int:
    """The shared Philox-4x32-10 word generator.

    Args:
        seed: 64-bit key (wrapped modulo 2**64).
        counter: 64-bit counter (wrapped modulo 2**64).

    Returns:
        The first word of the Philox-4x32-10 block, as a Python int in
        [0, 2**32).
    """
    seed &= (1 << 64) - 1
    counter &= (1 << 64) - 1
    k0 = np.uint32(seed & 0xFFFFFFFF)
    k1 = np.uint32(seed >> 32)
    x0 = np.uint32(counter & 0xFFFFFFFF)
    x1 = np.uint32(counter >> 32)
    x2 = np.uint32(seed & 0xFFFFFFFF)
    x3 = np.uint32(seed >> 32)
    for _ in range(10):
        hi0, lo0 = _mulhilo(_PHILOX_M0, x0)
        hi1, lo1 = _mulhilo(_PHILOX_M1, x2)
        x0, x1, x2, x3 = (
            np.uint32(hi1 ^ int(x1) ^ int(k0)),
            lo1,
            np.uint32(hi0 ^ int(x3) ^ int(k1)),
            lo0,
        )
        k0 = np.uint32((int(k0) + int(_PHILOX_W0)) & 0xFFFFFFFF)
        k1 = np.uint32((int(k1) + int(_PHILOX_W1)) & 0xFFFFFFFF)
    return int(x0)


def unit_float(word: int) -> np.float32:
    """Map a 32-bit word to float32 in [0, 1) using the top 24 bits."""
    return np.float32(np.float32(word >> 8) * np.float32(1.0 / 16777216.0))


def site_seed(network_seed: int, step: int, stream: int, key: int) -> int:
    """Derive the 64-bit per-site seed from the keying contract."""
    hi = philox32(network_seed, ((step << 8) | stream) & ((1 << 64) - 1))
    return ((hi << 32) | (key & 0xFFFFFFFF)) & ((1 << 64) - 1)


def conn_key(src: int, dst: int, occurrence: int) -> int:
    """Proposer key for a live connection."""
    return philox32((src << 32) | dst, occurrence)


class SiteRng:
    """Draw stream for one (site, proposal-index) pair.

    Each ``uniform`` / ``uniform_int`` call consumes one sub-counter slot.
    """

    def __init__(self, seed: int, j: int) -> None:
        self._seed = seed
        self._j = j
        self._sub = 0

    def _word(self) -> int:
        word = philox32(self._seed, ((self._j << 16) | self._sub))
        self._sub += 1
        return word

    def uniform(self) -> np.float32:
        """float32 in [0, 1)."""
        return unit_float(self._word())

    def uniform_int(self, n: int) -> int:
        """Integer in [0, n)."""
        u = self.uniform()
        return min(int(np.uint32(np.float32(u) * np.float32(n))), n - 1)


# ---------------------------------------------------------------------------
# State constructors
# ---------------------------------------------------------------------------


def make_unit(
    uid: int,
    level: int,
    *,
    pruned: bool = False,
    is_input: bool = False,
    is_output: bool = False,
    fields: dict[str, float] | None = None,
) -> dict[str, Any]:
    """One unit record."""
    return {
        "id": uid,
        "level": level,
        "pruned": pruned,
        "is_input": is_input,
        "is_output": is_output,
        "fields": dict(fields or {}),
    }


def make_edge(src: int, dst: int, **fields: float) -> dict[str, Any]:
    """One live edge record (slot order = list order)."""
    return {"src": src, "dst": dst, "fields": dict(fields)}


def live_units(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Units that are allocated and not pruned, ascending id."""
    return [u for u in units if not u["pruned"]]


def sorted_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Edges as the goldens record them: sorted, slot-agnostic."""
    return sorted(
        edges,
        key=lambda e: (e["src"], e["dst"], sorted(e["fields"].items())),
    )


# ---------------------------------------------------------------------------
# Passes
# ---------------------------------------------------------------------------


def forward_topological(
    units: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    inputs: list[float],
    apply_fn: Callable[[np.float32], np.float32],
) -> None:
    """Level-ordered forward sweep: act = apply(sum of w * act_src).

    Input units take ``inputs`` by ascending id; every deeper live unit
    accumulates its incoming edges (dead endpoints contribute nothing) in
    float32 and applies ``apply_fn``. Pruned units keep their activation.
    """
    in_ids = [u["id"] for u in units if u["is_input"]]
    for uid, value in zip(in_ids, inputs, strict=True):
        units[uid]["fields"]["activation"] = float(np.float32(value))
    levels = sorted({u["level"] for u in live_units(units) if not u["is_input"]})
    for level in levels:
        pending: dict[int, np.float32] = {}
        for u in live_units(units):
            if u["level"] != level or u["is_input"]:
                continue
            acc = np.float32(0.0)
            for e in edges:
                if e["dst"] != u["id"] or units[e["src"]]["pruned"]:
                    continue
                w = np.float32(e["fields"]["weight"])
                a = np.float32(units[e["src"]]["fields"]["activation"])
                acc = np.float32(acc + np.float32(w * a))
            pending[u["id"]] = apply_fn(acc)
        for uid, act in pending.items():
            units[uid]["fields"]["activation"] = float(act)


def forward_pipeline(
    units: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    inputs: list[float],
    apply_fn: Callable[[np.float32], np.float32],
) -> None:
    """One pipeline step: every non-input unit updates simultaneously.

    Messages read the activations from before this step, so a signal crosses
    exactly one connection per step.
    """
    old = {u["id"]: np.float32(u["fields"]["activation"]) for u in units}
    in_ids = [u["id"] for u in units if u["is_input"]]
    for uid, value in zip(in_ids, inputs, strict=True):
        units[uid]["fields"]["activation"] = float(np.float32(value))
        old[uid] = np.float32(value)
    for u in live_units(units):
        if u["is_input"]:
            continue
        acc = np.float32(0.0)
        for e in edges:
            if e["dst"] != u["id"] or units[e["src"]]["pruned"]:
                continue
            acc = np.float32(
                acc + np.float32(np.float32(e["fields"]["weight"]) * old[e["src"]])
            )
        u["fields"]["activation"] = float(apply_fn(acc))


def mse_loss_grad(units: list[dict[str, Any]], targets: list[float]) -> np.float32:
    """L = 0.5 * sum((act - t)^2) over outputs; stages grad = act - t.

    The staged gradient lands in the ``loss_grad`` field of each output unit.

    Returns:
        The scalar loss (reference-only; not asserted across libraries).
    """
    outs = [u for u in units if u["is_output"]]
    total = np.float32(0.0)
    for u, t in zip(outs, targets, strict=True):
        diff = np.float32(np.float32(u["fields"]["activation"]) - np.float32(t))
        u["fields"]["loss_grad"] = float(diff)
        total = np.float32(total + np.float32(0.5) * diff * diff)
    return total


def softmax_ce_loss_grad(
    units: list[dict[str, Any]],
    targets: list[float],
    *,
    log: Callable[[np.float32], np.float32] = np.log,
) -> np.float32:
    """Softmax cross-entropy over the outputs; stages the seed into loss_grad.

    The float32 operation order is the specification (module docstring,
    Loss). ``log`` exists so the emitter can prove a golden insensitive to
    the last bits of ``log(s)``, which libraries need not round identically.

    Returns:
        The scalar loss.
    """
    outs = [u for u in units if u["is_output"]]
    acts = [np.float32(u["fields"]["activation"]) for u in outs]
    peak = np.float32(-np.inf)
    for a in acts:
        if a > peak:
            peak = a
    total_exp = np.float32(0.0)
    for a in acts:
        total_exp = np.float32(total_exp + np.exp(np.float32(a - peak)))
    log_sum = np.float32(log(total_exp))
    total = np.float32(0.0)
    for u, a, t in zip(outs, acts, targets, strict=True):
        z = np.float32(a - peak)
        t32 = np.float32(t)
        seed = np.float32(np.float32(np.exp(z) / total_exp) - t32)
        u["fields"]["loss_grad"] = float(seed)
        total = np.float32(total + np.float32(t32 * np.float32(log_sum - z)))
    return total


def backward_topological(
    units: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    dprime: Callable[[np.float32], np.float32],
) -> None:
    """Reverse level walk: grad = (acc + loss_grad) * dprime(activation).

    ``acc`` sums ``w * grad_pre_act[dst]`` over the unit's outgoing edges to
    deeper units; ``loss_grad`` is non-zero only on outputs. The result is
    written to ``grad_pre_act``. Input units are not applied: both libraries
    run backward Apply on non-input units only, so an input's ``grad_pre_act``
    keeps its default.
    """
    levels = sorted({u["level"] for u in live_units(units)}, reverse=True)
    for level in levels:
        pending: dict[int, np.float32] = {}
        for u in live_units(units):
            if u["level"] != level or u["is_input"]:
                continue
            acc = np.float32(0.0)
            for e in edges:
                if e["src"] != u["id"] or units[e["dst"]]["pruned"]:
                    continue
                w = np.float32(e["fields"]["weight"])
                g = np.float32(units[e["dst"]]["fields"].get("grad_pre_act", 0.0))
                acc = np.float32(acc + np.float32(w * g))
            seed = np.float32(u["fields"].get("loss_grad", 0.0))
            a = np.float32(u["fields"]["activation"])
            pending[u["id"]] = np.float32(np.float32(acc + seed) * dprime(a))
        for uid, grad in pending.items():
            units[uid]["fields"]["grad_pre_act"] = float(grad)


def relu(acc: np.float32) -> np.float32:
    """max(acc, 0)."""
    return acc if acc > np.float32(0.0) else np.float32(0.0)


def relu_prime_from_act(act: np.float32) -> np.float32:
    """1 where the activation is positive, else 0."""
    return np.float32(1.0) if act > np.float32(0.0) else np.float32(0.0)


# ---------------------------------------------------------------------------
# Unit lifecycle
# ---------------------------------------------------------------------------


def update_units(
    units: list[dict[str, Any]],
    rule: Callable[[dict[str, Any]], dict[str, float]],
) -> None:
    """Apply ``rule`` to every live unit (inputs and outputs included)."""
    for u in live_units(units):
        u["fields"].update(rule(u))


def prune_units(
    units: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
    field_defaults: dict[str, float],
) -> list[int]:
    """Permanently prune live non-input, non-output units.

    The predicate is evaluated on the pre-phase state of every eligible unit;
    every unit it selects is marked pruned, its incident edges are removed,
    and its fields reset to their declared defaults.

    Returns:
        The pruned ids, ascending.
    """
    doomed = [
        u["id"]
        for u in live_units(units)
        if not u["is_input"] and not u["is_output"] and predicate(u)
    ]
    for uid in doomed:
        units[uid]["pruned"] = True
        units[uid]["fields"] = dict(field_defaults)
    edges[:] = [
        e for e in edges if e["src"] not in set(doomed) and e["dst"] not in set(doomed)
    ]
    return doomed


def add_units(
    units: list[dict[str, Any]],
    *,
    capacity: int,
    max_levels: int,
    spawn: Callable[[dict[str, Any]], tuple[bool, int]],
    init_child: Callable[[dict[str, Any], dict[str, Any]], dict[str, float]],
    field_defaults: dict[str, float],
) -> tuple[list[int], bool]:
    """Spawn children per the addition semantics in the module docstring.

    Args:
        units: Unit records; mutated in place. ``len(units)`` is the
            allocation high-water mark.
        capacity: Total unit capacity.
        max_levels: Level clamp bound (children land in [1, max_levels - 1]).
        spawn: Maps a parent to ``(spawn?, level offset)``.
        init_child: Maps ``(child, parent)`` to the fields the rule writes.
        field_defaults: Values for fields the rule leaves unwritten.

    Returns:
        ``(new child ids in assignment order, unit_overflow)``.
    """
    parents = [u for u in live_units(units)]  # pre-phase snapshot
    free = [u["id"] for u in units if u["pruned"]] + list(range(len(units), capacity))
    free.sort()
    children: list[int] = []
    overflow = False
    for parent in parents:
        wants, offset = spawn(parent)
        if not wants:
            continue
        if not free:
            overflow = True
            continue
        cid = free.pop(0)
        level = max(1, min(parent["level"] + offset, max_levels - 1))
        child = make_unit(cid, level, fields=dict(field_defaults))
        child["fields"].update(init_child(child, parent))
        if cid < len(units):
            units[cid] = child
        else:
            while len(units) < cid:
                units.append(make_unit(len(units), 0, pruned=True))
            units.append(child)
        children.append(cid)
    return children, overflow


# ---------------------------------------------------------------------------
# Growth
# ---------------------------------------------------------------------------

NEG_INF = float("-inf")


def _occurrences(edges: list[dict[str, Any]]) -> list[tuple[int, int, int]]:
    """(src, dst, occurrence) per live edge, occurrence in slot order."""
    seen: dict[tuple[int, int], int] = {}
    out = []
    for e in edges:
        key = (e["src"], e["dst"])
        occ = seen.get(key, 0)
        seen[key] = occ + 1
        out.append((e["src"], e["dst"], occ))
    return out


def candidates_per_unit(
    units: list[dict[str, Any]],
    *,
    proposals: int,
    network_seed: int,
    step: int,
    propose: Callable[[dict[str, Any], int, SiteRng], tuple[int, int, float]],
) -> list[dict[str, Any]]:
    """Per-unit proposals: live units x P, index = unit_id * P + j."""
    out = []
    for u in live_units(units):
        seed = site_seed(network_seed, step, 1, u["id"])
        for j in range(proposals):
            rng = SiteRng(seed, j)
            src, dst, score = propose(u, j, rng)
            out.append(
                {
                    "src": src,
                    "dst": dst,
                    "score": score,
                    "index": u["id"] * proposals + j,
                }
            )
    return out


def candidates_per_connection(
    units: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    proposals: int,
    network_seed: int,
    step: int,
    propose: Callable[
        [dict[str, Any], tuple[int, int, int], int, SiteRng], tuple[int, int, float]
    ],
) -> list[dict[str, Any]]:
    """Per-connection proposals: index = r * P + j, r the (src, dst, occ) rank."""
    occs = _occurrences(edges)
    ranked = sorted(range(len(edges)), key=lambda i: occs[i])
    out = []
    for r, ei in enumerate(ranked):
        src, dst, occ = occs[ei]
        if units[src]["pruned"] or units[dst]["pruned"]:
            continue
        seed = site_seed(network_seed, step, 1, conn_key(src, dst, occ))
        for j in range(proposals):
            rng = SiteRng(seed, j)
            psrc, pdst, score = propose(edges[ei], (src, dst, occ), j, rng)
            out.append(
                {"src": psrc, "dst": pdst, "score": score, "index": r * proposals + j}
            )
    return out


def candidates_global(
    *,
    proposals: int,
    network_seed: int,
    step: int,
    propose: Callable[[int, SiteRng], tuple[int, int, float]],
) -> list[dict[str, Any]]:
    """Global proposals: P candidates, index = j."""
    out = []
    seed = site_seed(network_seed, step, 1, 0)
    for j in range(proposals):
        rng = SiteRng(seed, j)
        src, dst, score = propose(j, rng)
        out.append({"src": src, "dst": dst, "score": score, "index": j})
    return out


def candidates_exhaustive(
    units: list[dict[str, Any]],
    *,
    capacity: int,
    score: Callable[[int, int], float],
) -> list[dict[str, Any]]:
    """Every live ordered pair, index = src * capacity + dst."""
    out = []
    for s in live_units(units):
        for d in live_units(units):
            out.append(
                {
                    "src": s["id"],
                    "dst": d["id"],
                    "score": score(s["id"], d["id"]),
                    "index": s["id"] * capacity + d["id"],
                }
            )
    return out


def candidates_shortlist(
    units: list[dict[str, Any]],
    *,
    shortlist_size: int,
    importance: Callable[[int], float],
    score: Callable[[int, int], float],
) -> list[dict[str, Any]]:
    """The M x M grid over the importance-ranked units, row-major index.

    Importance ties break by ascending unit id.
    """
    ranked = sorted(live_units(units), key=lambda u: (-importance(u["id"]), u["id"]))[
        :shortlist_size
    ]
    out = []
    for i, s in enumerate(ranked):
        for k, d in enumerate(ranked):
            out.append(
                {
                    "src": s["id"],
                    "dst": d["id"],
                    "score": score(s["id"], d["id"]),
                    "index": i * len(ranked) + k,
                }
            )
    return out


def candidates_shortlist_per_level(
    units: list[dict[str, Any]],
    *,
    shortlist_size: int,
    max_level_gap: int,
    direction: str = "any",
    importance: Callable[[int], float],
    score: Callable[[int, int], float],
) -> list[dict[str, Any]]:
    """One M x M grid per source level, levels ascending.

    Sources are the level's top-M live units by importance; destinations are
    the top-M live units inside the level's validity window (level gap and
    direction), by importance. Importance ties break by ascending unit id.
    Index = level_rank * M * M + row-major position in the level's grid.
    """

    def ranked(eligible: Callable[[dict[str, Any]], bool]) -> list[dict[str, Any]]:
        pool = [u for u in live_units(units) if eligible(u)]
        return sorted(pool, key=lambda u: (-importance(u["id"]), u["id"]))[
            :shortlist_size
        ]

    def in_window(src_level: int, dst_level: int) -> bool:
        if abs(dst_level - src_level) > max_level_gap:
            return False
        if direction == "deeper":
            return dst_level > src_level
        if direction == "same_or_deeper":
            return dst_level >= src_level
        return True

    levels = sorted({u["level"] for u in live_units(units)})
    m = shortlist_size
    out = []
    for rank, level in enumerate(levels):
        srcs = ranked(lambda u, lv=level: u["level"] == lv)
        dsts = ranked(lambda u, lv=level: in_window(lv, u["level"]))
        for i, s in enumerate(srcs):
            for k, d in enumerate(dsts):
                out.append(
                    {
                        "src": s["id"],
                        "dst": d["id"],
                        "score": score(s["id"], d["id"]),
                        "index": rank * m * m + i * len(dsts) + k,
                    }
                )
    return out


def select_growth(
    units: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    cands: list[dict[str, Any]],
    *,
    capacity: int,
    max_level_gap: int = 1,
    direction: str = "any",
    allow_self_loops: bool = False,
    dedupe_live: bool = False,
    dedupe_step: bool = False,
    selection: str = "top_k",
    max_new_per_level: int | None = None,
    max_new_per_step: int | None = None,
    threshold: float | None = None,
    topological: bool = True,
) -> list[dict[str, Any]]:
    """Stages 3-7 of the growth pipeline: validity through selection.

    Returns:
        The selected candidates in commit order (before slot claiming).
    """
    for c in cands:
        score = c["score"]
        valid = (
            0 <= c["src"] < capacity
            and 0 <= c["dst"] < capacity
            and c["src"] < len(units)
            and c["dst"] < len(units)
            and not units[c["src"]]["pruned"]
            and not units[c["dst"]]["pruned"]
        )
        if valid and not allow_self_loops and c["src"] == c["dst"]:
            valid = False
        if valid:
            ls = units[c["src"]]["level"]
            ld = units[c["dst"]]["level"]
            if abs(ld - ls) > max_level_gap:
                valid = False
            elif direction == "deeper" and not ld > ls:
                valid = False
            elif direction == "same_or_deeper" and not ld >= ls:
                valid = False
        if not valid or not math.isfinite(score):
            c["score"] = NEG_INF

    if dedupe_live:
        live = {(e["src"], e["dst"]) for e in edges}
        for c in cands:
            if (c["src"], c["dst"]) in live:
                c["score"] = NEG_INF

    order = sorted(cands, key=lambda c: (-c["score"], c["src"], c["dst"], c["index"]))

    if dedupe_step:
        seen: set[tuple[int, int]] = set()
        for c in order:
            key = (c["src"], c["dst"])
            if c["score"] == NEG_INF:
                continue
            if key in seen:
                c["score"] = NEG_INF
            else:
                seen.add(key)
        order = sorted(
            cands, key=lambda c: (-c["score"], c["src"], c["dst"], c["index"])
        )

    finite = [c for c in order if c["score"] != NEG_INF]
    per_level: dict[int, list[dict[str, Any]]] = {}
    for c in finite:
        per_level.setdefault(units[c["src"]]["level"], []).append(c)

    selected: list[dict[str, Any]] = []
    for level in sorted(per_level):
        bucket = per_level[level]
        if selection == "top_k":
            assert max_new_per_level is not None
            take = bucket[:max_new_per_level]
        elif selection == "threshold":
            assert threshold is not None and max_new_per_level is not None
            take = [c for c in bucket if c["score"] >= threshold][:max_new_per_level]
        elif selection == "all":
            take = bucket
        else:  # pragma: no cover - emit-time misuse
            raise ValueError(f"unknown selection {selection!r}")
        selected.extend(take)
    if max_new_per_step is not None:
        selected = selected[:max_new_per_step]
    del topological  # bucket gating is folded into validity by the callers
    return selected


def commit_growth(
    units: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    *,
    init: Callable[[int, int], dict[str, float]],
    field_defaults: dict[str, float],
    free_slots: int | None = None,
    dead_per_level: dict[int, int] | None = None,
    tail: int | None = None,
    free_per_level: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Stages 8-10: claim slots, init fields, compute flags.

    Exactly one claim model is given:

    - ``free_slots``: one domain of that many never-used slots (the pipeline
      model with no dead slots), committed as a prefix of the selection;
    - ``dead_per_level`` and ``tail``: the pipeline model -- each level's own
      dead slots first, then the tail, levels ascending;
    - ``free_per_level``: the topological model -- each level only its own
      bucket's free slots. The result also lists ``regrow_levels``, the
      levels whose bucket the claim left full, ascending.

    ``committed`` keeps the selection order (levels ascending, the total order
    within a level).
    """
    level_of = {u["id"]: u["level"] for u in units}
    by_level: dict[int, list[dict[str, Any]]] = {}
    for c in selected:
        by_level.setdefault(level_of[c["src"]], []).append(c)
    extra: dict[str, Any] = {}
    if free_slots is not None:
        assert dead_per_level is None and tail is None and free_per_level is None
        keep = {id(c) for c in selected[:free_slots]}
    elif free_per_level is not None:
        assert dead_per_level is None and tail is None
        keep = set()
        left = dict(free_per_level)
        for level in sorted(by_level):
            assert level in left, f"no bucket for source level {level}"
            take = by_level[level][: left[level]]
            keep |= {id(c) for c in take}
            left[level] -= len(take)
        extra["regrow_levels"] = sorted(lvl for lvl, f in left.items() if f == 0)
    else:
        assert dead_per_level is not None and tail is not None
        keep = set()
        spill: list[dict[str, Any]] = []
        for level in sorted(by_level):
            own = dead_per_level.get(level, 0)
            keep |= {id(c) for c in by_level[level][:own]}
            spill.extend(by_level[level][own:])
        keep |= {id(c) for c in spill[:tail]}
        extra["tail_used"] = min(len(spill), tail)
    committed = [c for c in selected if id(c) in keep]
    overflow = len(selected) > len(committed)
    for c in committed:
        fields = dict(field_defaults)
        fields.update(init(c["src"], c["dst"]))
        edges.append(make_edge(c["src"], c["dst"], **fields))
    needs_resort = any(
        units[c["dst"]]["level"] <= units[c["src"]]["level"] for c in committed
    )
    return {
        "committed": [
            {"src": c["src"], "dst": c["dst"], "score": c["score"]} for c in committed
        ],
        "grown": len(committed),
        "conn_overflow": overflow,
        "needs_resort": needs_resort,
        **extra,
    }
