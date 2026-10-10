"""Emit the registry goldens from the pure-NumPy reference.

Writes every golden to this repository's ``tests/golden/`` and, when a
plastax-cpp checkout is reachable (``$PLASTAX_CPP_DIR``, falling back to a
``plastax-cpp`` directory beside this repository), to its ``tests/golden/``
too. ``--check`` regenerates in memory and fails if any committed golden
differs byte-for-byte -- goldens change only by deliberately re-running this
script and committing the diff.

Golden families, by their ``requires`` tag:

- ``passes_v1``: forward/backward/MSE on a fixed dyadic ReLU MLP. Consumed by
  both libraries today, with exact float equality.
- ``loss_v1``: the whole-output loss contract, through softmax
  cross-entropy on a linear identity layer: the scalar loss and every output's
  gradient seed. Consumed by both libraries, with exact float equality.
- ``unit_lifecycle_v1``: the unit update/prune/add semantics. Spec-only until
  the libraries implement the unit lifecycle; consumers skip them loudly.
- ``growth_v2``: the shared growth selection pipeline. Spec-only until the
  libraries implement it; consumers skip them loudly.

These are distinct from the network conformance vectors
(``scripts/emit_parity_goldens.py``), which pin whole-network trajectories
with tolerances; registry goldens pin individual phase semantics exactly.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import sys
from typing import Any

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "tests"))

import golden_rules as rules  # noqa: E402
import reference as ref  # noqa: E402

_REPO = _HERE.parents[1]
_PX_GOLDEN = _REPO / "tests" / "golden"


def _cx_golden_dir() -> pathlib.Path | None:
    env = os.environ.get("PLASTAX_CPP_DIR")
    root = (
        pathlib.Path(env).expanduser().resolve()
        if env
        else _REPO.parent / "plastax-cpp"
    )
    return root / "tests" / "golden" if root.is_dir() else None


def _doc(name: str, requires: str, comment: str, **payload: Any) -> dict[str, Any]:
    return {
        "_comment": comment,
        "schema": "registry_v1",
        "generator": "scripts/parity/emit.py",
        "name": name,
        "requires": requires,
        **payload,
    }


def _units_json(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return copy.deepcopy(units)


def _edges_json(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return copy.deepcopy(ref.sorted_edges(edges))


# ---------------------------------------------------------------------------
# passes_v1
# ---------------------------------------------------------------------------

_PASS_LAYERS = (3, 2)
_PASS_INPUTS = 2


def _mlp_state() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The fixed dyadic ReLU MLP: 2 inputs, layers of 3 and 2 units."""
    units: list[dict[str, Any]] = []
    uid = 0
    for _ in range(_PASS_INPUTS):
        units.append(ref.make_unit(uid, 0, is_input=True, fields={"activation": 0.0}))
        uid += 1
    prev = list(range(_PASS_INPUTS))
    for li, width in enumerate(_PASS_LAYERS):
        layer = []
        for _ in range(width):
            is_out = li == len(_PASS_LAYERS) - 1
            units.append(
                ref.make_unit(
                    uid,
                    li + 1,
                    is_output=is_out,
                    fields={"activation": 0.0, "grad_pre_act": 0.0, "loss_grad": 0.0},
                )
            )
            layer.append(uid)
            uid += 1
        prev, layer_src = layer, prev
        del layer_src
    edges = []
    # Fully connected, destination-major within a layer (both builders agree).
    starts = [0, _PASS_INPUTS]
    for width in _PASS_LAYERS:
        starts.append(starts[-1] + width)
    for li, width in enumerate(_PASS_LAYERS):
        srcs = range(starts[li], starts[li + 1])
        dsts = range(starts[li + 1], starts[li + 1] + width)
        for d in dsts:
            for s in srcs:
                edges.append(ref.make_edge(s, d, weight=rules.dyadic_weight_v1(s, d)))
    return units, edges


_PASS_STEPS = (
    ([1.0, -0.5], [0.25, -0.75]),
    ([0.5, 0.5], [1.0, 0.0]),
    ([-1.0, 2.0], [0.5, 0.5]),
)


def _passes_topological() -> dict[str, Any]:
    units, edges = _mlp_state()
    steps = []
    for inputs, targets in _PASS_STEPS:
        ref.forward_topological(units, edges, inputs, ref.relu)
        loss = ref.mse_loss_grad(units, targets)
        ref.backward_topological(units, edges, ref.relu_prime_from_act)
        steps.append(
            {
                "inputs": inputs,
                "targets": targets,
                "expect": {
                    "activations": [u["fields"]["activation"] for u in units],
                    "grad_pre_act": [
                        u["fields"].get("grad_pre_act", 0.0) for u in units
                    ],
                    "loss": float(loss),
                },
            }
        )
    return _doc(
        "passes_relu_topological",
        "passes_v1",
        "Forward (ReLU), MSE loss and backward on a fixed dyadic MLP, "
        "topological mode. All values are dyadic; compare exactly. The loss "
        "value is reference-only. Weights are dyadic_weight_v1 over global "
        "unit ids; grad_pre_act = (sum_out w*grad_dst + loss_grad) * [act>0].",
        mode="topological",
        rules={"passes": rules.RELU_MLP_V1, "weights": "dyadic_weight_v1"},
        network={"input_dim": _PASS_INPUTS, "layers": list(_PASS_LAYERS)},
        initial_edges=_edges_json(_mlp_state()[1]),
        steps=steps,
    )


def _passes_pipeline() -> dict[str, Any]:
    units, edges = _mlp_state()
    steps = []
    for inputs in ([1.0, -0.5], [0.5, 0.5], [-1.0, 2.0], [0.25, 0.25]):
        ref.forward_pipeline(units, edges, inputs, ref.relu)
        steps.append(
            {
                "inputs": inputs,
                "expect": {"activations": [u["fields"]["activation"] for u in units]},
            }
        )
    return _doc(
        "passes_relu_pipeline",
        "passes_v1",
        "Forward-only pipeline mode on the same dyadic MLP: every non-input "
        "unit updates simultaneously from the previous step's activations, so "
        "a signal crosses one connection per step. Compare exactly.",
        mode="pipeline",
        rules={"passes": rules.RELU_MLP_V1, "weights": "dyadic_weight_v1"},
        network={"input_dim": _PASS_INPUTS, "layers": list(_PASS_LAYERS)},
        initial_edges=_edges_json(_mlp_state()[1]),
        steps=steps,
    )


# ---------------------------------------------------------------------------
# loss_v1
# ---------------------------------------------------------------------------

_LOSS_WIDTH = 4

# (name, comment, steps of (logits, targets)). Every shifted logit is 0 or
# <= -104, so each exp is exactly 1 or 0 in every float32 implementation, and
# the exponential sum is the (power-of-two) number of tied maxima.
_LossStep = tuple[list[float], list[float]]
_LOSS_CASES: tuple[tuple[str, str, tuple[_LossStep, ...]], ...] = (
    (
        "loss_softmax_ce_one_hot",
        "One-hot targets, on and off the argmax: the seed is softmax - target "
        "on every output and the loss is the target's log-sum-exp margin.",
        (
            ([3.0, -125.0, -130.0, -141.0], [0.0, 0.0, 1.0, 0.0]),
            ([-128.0, 0.0, -136.0, -200.0], [0.0, 1.0, 0.0, 0.0]),
            ([-110.0, -250.0, 4.0, -112.0], [1.0, 0.0, 0.0, 0.0]),
        ),
    ),
    (
        "loss_softmax_ce_soft_targets",
        "A soft (dyadic) target distribution: the loss weights every "
        "output's log-sum-exp margin by its target mass.",
        (
            ([-2.0, -130.0, -134.0, -150.0], [0.25, 0.5, 0.125, 0.125]),
            ([-160.0, -140.0, 12.0, -120.0], [0.5, 0.0, 0.25, 0.25]),
        ),
    ),
    (
        "loss_softmax_ce_large_logits",
        "Logits whose unshifted exponentials overflow float32: only the "
        "max-subtracted form stays finite.",
        (
            ([300.0, 150.0, 180.0, 64.0], [0.0, 0.5, 0.5, 0.0]),
            ([96.0, 224.0, 88.0, 100.0], [0.0, 0.0, 0.0, 1.0]),
        ),
    ),
    (
        "loss_softmax_ce_negative_logits",
        "Logits whose unshifted exponentials all underflow to 0: only the "
        "max-subtracted form avoids 0 / 0.",
        (
            ([-420.0, -300.0, -500.0, -410.0], [0.25, 0.0, 0.25, 0.5]),
            ([-250.0, -480.0, -360.0, -380.0], [0.0, 1.0, 0.0, 0.0]),
        ),
    ),
    (
        "loss_softmax_ce_tied_max",
        "Two tied maxima share the softmax mass (exactly 1/2 each), so the "
        "seed exercises the division by the exponential sum; the loss adds "
        "log(2) to a margin large enough to absorb its last bits.",
        (
            ([5.0, 5.0, -128.0, -140.0], [0.0, 0.0, 1.0, 0.0]),
            ([-300.0, 40.0, -100.0, 40.0], [0.0, 0.0, 1.0, 0.0]),
        ),
    ),
)


def _loss_state() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``_LOSS_WIDTH`` inputs, each copied to one output (identity weights)."""
    n = _LOSS_WIDTH
    units = [
        ref.make_unit(i, 0, is_input=True, fields={"activation": 0.0}) for i in range(n)
    ]
    units += [
        ref.make_unit(n + k, 1, is_output=True, fields={"activation": 0.0})
        for k in range(n)
    ]
    edges = [
        ref.make_edge(s, n + d, weight=rules.identity_weight_v1(s, n + d, n))
        for d in range(n)
        for s in range(n)
    ]
    return units, edges


def _check_exact(units: list[dict[str, Any]], targets: list[float]) -> np.float32:
    """Run the reference, asserting the golden is exact by construction.

    Every shifted logit must be 0 or underflow (so exp is exact everywhere),
    the tie count a power of two (so the softmax quotient is exact), and the
    loss unchanged by +-64 ulp in log(s) or by reversing the summation. A
    single maximum makes s = 1, whose logarithm is exactly 0 everywhere.
    """
    acts = [np.float32(u["fields"]["activation"]) for u in units if u["is_output"]]
    peak = max(acts)
    shifted = [np.float32(a - peak) for a in acts]
    assert all(z == 0 or z <= -104 for z in shifted), shifted
    assert sum(z == 0 for z in shifted) in (1, 2, 4), shifted

    def nudged(ulps: int) -> Any:
        def log(x: np.float32) -> np.float32:
            v = np.float32(np.log(x))
            if x == 1:
                return v
            step = np.float32(np.inf if ulps > 0 else -np.inf)
            for _ in range(abs(ulps)):
                v = np.nextafter(v, step, dtype=np.float32)
            return v

        return log

    loss = ref.softmax_ce_loss_grad(units, targets)
    for ulps in (-64, 64):
        probe = copy.deepcopy(units)
        assert ref.softmax_ce_loss_grad(probe, targets, log=nudged(ulps)) == loss
    flipped = copy.deepcopy(units)
    outs = [u for u in flipped if u["is_output"]]
    for u, a in zip(outs, reversed(acts), strict=True):
        u["fields"]["activation"] = float(a)
    assert ref.softmax_ce_loss_grad(flipped, targets[::-1]) == loss
    return loss


def _loss_cases() -> list[dict[str, Any]]:
    docs = []
    for name, comment, case_steps in _LOSS_CASES:
        units, edges = _loss_state()
        steps = []
        for logits, targets in case_steps:
            ref.forward_topological(units, edges, logits, lambda acc: acc)
            loss = _check_exact(units, targets)
            outs = [u for u in units if u["is_output"]]
            steps.append(
                {
                    "inputs": logits,
                    "targets": targets,
                    "expect": {
                        "activations": [u["fields"]["activation"] for u in outs],
                        "seed": [u["fields"]["loss_grad"] for u in outs],
                        "loss": float(loss),
                    },
                }
            )
        docs.append(
            _doc(
                name,
                "loss_v1",
                comment + " Compare exactly: the seed is the loss's gradient "
                "seed on each output (in output order), the loss the scalar it "
                "returns. Forward is linear_v1 over identity_weight_v1, so each "
                "output's activation is its input logit.",
                rules={
                    "loss": rules.SOFTMAX_CE_V1,
                    "forward": rules.LINEAR_V1,
                    "weights": "identity_weight_v1",
                },
                network={"input_dim": _LOSS_WIDTH, "layers": [_LOSS_WIDTH]},
                initial_edges=_edges_json(edges),
                steps=steps,
            )
        )
    return docs


# ---------------------------------------------------------------------------
# unit_lifecycle_v1
# ---------------------------------------------------------------------------

_UNIT_DEFAULTS = {"activation": 0.0}


def _lifecycle_state() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """6 allocated units of capacity 8: in(0), hidden(1..4), out(5)."""
    units = [
        ref.make_unit(0, 0, is_input=True, fields={"activation": 0.25}),
        ref.make_unit(1, 1, fields={"activation": -0.75}),
        ref.make_unit(2, 1, fields={"activation": 0.5}),
        ref.make_unit(3, 2, pruned=True, fields=dict(_UNIT_DEFAULTS)),
        ref.make_unit(4, 2, fields={"activation": 0.75}),
        ref.make_unit(5, 3, is_output=True, fields={"activation": -0.875}),
    ]
    edges = [
        ref.make_edge(0, 1, weight=0.5),
        ref.make_edge(0, 2, weight=-0.25),
        ref.make_edge(1, 4, weight=0.125),
        ref.make_edge(2, 4, weight=0.25),
        ref.make_edge(4, 5, weight=-0.5),
    ]
    return units, edges


def _unit_case(
    name: str,
    comment: str,
    *,
    capacity: int = 8,
    max_levels: int = 1024,
    do_update: bool = False,
    do_prune: bool = False,
    spawn: str | None = None,
) -> dict[str, Any]:
    units, edges = _lifecycle_state()
    initial_units = _units_json(units)
    initial_edges = _edges_json(edges)
    expect: dict[str, Any] = {}
    if do_update:
        ref.update_units(units, rules.unit_update_v1)
    pruned: list[int] = []
    if do_prune:
        pruned = ref.prune_units(units, edges, rules.unit_prune_v1, _UNIT_DEFAULTS)
    if spawn is not None:
        spawn_rule = (
            rules.unit_spawn_deep_v1 if spawn == "deep" else rules.unit_spawn_v1
        )
        children, overflow = ref.add_units(
            units,
            capacity=capacity,
            max_levels=max_levels,
            spawn=spawn_rule,
            init_child=rules.unit_child_init_v1,
            field_defaults=_UNIT_DEFAULTS,
        )
        expect["children"] = children
        expect["unit_overflow"] = overflow
    expect["units"] = _units_json(units)
    expect["edges"] = _edges_json(edges)
    expect["pruned"] = pruned
    doc_rules = {}
    if do_update:
        doc_rules["update_unit"] = "unit_update_v1"
    if do_prune:
        doc_rules["prune_unit"] = "unit_prune_v1"
    if spawn is not None:
        doc_rules["add_unit"] = (
            "unit_spawn_deep_v1" if spawn == "deep" else "unit_spawn_v1"
        )
        doc_rules["init_unit"] = "unit_child_init_v1"
    return _doc(
        name,
        "unit_lifecycle_v1",
        comment,
        capacity=capacity,
        max_levels=max_levels,
        field_defaults=_UNIT_DEFAULTS,
        rules=doc_rules,
        initial_units=initial_units,
        initial_edges=initial_edges,
        expect=expect,
    )


def _unit_cases() -> list[dict[str, Any]]:
    cases = [
        _unit_case(
            "unit_update_basic",
            "update_unit runs on every live unit, inputs and outputs included.",
            do_update=True,
        ),
        _unit_case(
            "unit_prune_permanent",
            "Pruning is permanent: unit 1 (activation -0.75 < -1/2) is pruned, "
            "its incident edges die in the same phase, its fields reset to "
            "declared defaults. The never-allocated slots stay free.",
            do_prune=True,
        ),
        _unit_case(
            "unit_prune_exempt_io",
            "Input and output units are exempt from pruning even when the "
            "predicate would select them: unit 5 (output, activation -0.875) "
            "survives; only hidden unit 1 is pruned.",
            do_prune=True,
        ),
        _unit_case(
            "unit_add_lowest_free_id",
            "Units 2 and 4 spawn (activation >= 1/2); free ids are the pruned "
            "id 3 then the never-allocated 6 and 7, so the first spawner takes "
            "3 and the second takes 6. Children do not spawn this step.",
            spawn="basic",
        ),
        _unit_case(
            "unit_add_reuse_same_step",
            "Prune then add in one step: unit 1 is pruned by this step's prune "
            "phase and its id is already reusable when the add phase assigns "
            "ids, so the spawners receive 1 and 3 (the lowest free ids).",
            do_prune=True,
            spawn="basic",
        ),
        _unit_case(
            "unit_add_overflow",
            "Capacity 6 means the only free id is pruned slot 3; the first "
            "spawner (unit 2) takes it and the second (unit 4) is dropped, "
            "raising unit_overflow.",
            capacity=6,
            spawn="basic",
        ),
        _unit_case(
            "unit_add_level_clamp",
            "A +30000 level offset clamps to max_levels - 1.",
            max_levels=16,
            spawn="deep",
        ),
    ]
    return cases


# ---------------------------------------------------------------------------
# growth_v2
# ---------------------------------------------------------------------------

_GROW_SEED = 7
_GROW_STEP = 3
_GROW_CAPACITY = 16  # unit-id validity bound in the shared growth net


def _growth_state(
    *, isolated_unit: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """6 units, levels [0,0,1,1,2,2]; a parallel edge exercises occurrences."""
    units = [
        ref.make_unit(0, 0, is_input=True, fields={"activation": 0.5}),
        ref.make_unit(1, 0, is_input=True, fields={"activation": -0.25}),
        ref.make_unit(2, 1, fields={"activation": 0.125}),
        ref.make_unit(3, 1, fields={"activation": 0.75}),
        ref.make_unit(4, 2, is_output=True, fields={"activation": -0.5}),
        ref.make_unit(5, 2, is_output=True, fields={"activation": 0.25}),
    ]
    edges = [
        ref.make_edge(0, 2, weight=0.5),
        ref.make_edge(1, 3, weight=-0.5),
        ref.make_edge(2, 4, weight=0.25),
        ref.make_edge(2, 4, weight=0.125),  # parallel: occurrence 1
        ref.make_edge(3, 5, weight=-0.25),
    ]
    if isolated_unit:
        units.append(ref.make_unit(6, 1, fields={"activation": 1.0}))
    return units, edges


def _per_level_state() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """9 units, three per level, so each level's top-M (M=2) is a real choice.

    importance_v1 ranks level 0 as 1 > 2 > 0, level 1 as 5 > 3 > 4 and
    level 2 as 7 > 8 > 6.
    """
    units = [
        ref.make_unit(i, 0, is_input=True, fields={"activation": 0.25 * i})
        for i in range(3)
    ]
    units += [
        ref.make_unit(3 + i, 1, fields={"activation": 0.5 - 0.25 * i}) for i in range(3)
    ]
    units += [
        ref.make_unit(6 + i, 2, is_output=True, fields={"activation": 0.125 * (1 - i)})
        for i in range(3)
    ]
    edges = [
        ref.make_edge(0, 3, weight=0.5),
        ref.make_edge(1, 4, weight=-0.25),
        ref.make_edge(2, 5, weight=0.75),
        ref.make_edge(1, 3, weight=0.125),
        ref.make_edge(3, 6, weight=-0.5),
        ref.make_edge(4, 7, weight=0.25),
        ref.make_edge(5, 8, weight=1.0),
    ]
    return units, edges


def _grow_doc(
    name: str,
    comment: str,
    *,
    cands: list[dict[str, Any]],
    units: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    params: dict[str, Any],
    rules_used: dict[str, str],
    free_slots: int = 8,
    initial_units: list[dict[str, Any]] | None = None,
    initial_edges: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    select_args = dict(params)
    seed_step = {
        "network_seed": select_args.pop("network_seed", _GROW_SEED),
        "step": select_args.pop("step", _GROW_STEP),
    }
    # Descriptive-only parameters: recorded in the golden, not select stages.
    descriptive = {
        k: select_args.pop(k) for k in ("proposer", "candidates") if k in select_args
    }
    proposals = select_args.pop("proposals_per_proposer", None)
    shortlist = select_args.pop("shortlist_size", None)
    selected = ref.select_growth(
        units, edges, cands, capacity=_GROW_CAPACITY, **select_args
    )
    outcome = ref.commit_growth(
        units,
        edges,
        selected,
        free_slots=free_slots,
        init=rules.grow_init_v1,
        field_defaults={"weight": 0.0},
    )
    doc_params = {
        "capacity": _GROW_CAPACITY,
        "free_slots": free_slots,
        **seed_step,
        **descriptive,
        **select_args,
    }
    if proposals is not None:
        doc_params["proposals_per_proposer"] = proposals
    if shortlist is not None:
        doc_params["shortlist_size"] = shortlist
    return _doc(
        name,
        "growth_v2",
        comment,
        rules=rules_used,
        params=doc_params,
        initial_units=initial_units
        if initial_units is not None
        else _units_json(units),
        initial_edges=initial_edges
        if initial_edges is not None
        else _edges_json([e for e in edges if "committed" not in e]),
        expect=outcome,
    )


def _grow_cases() -> list[dict[str, Any]]:
    n = 7  # covers the isolated-unit variant's id range
    cases: list[dict[str, Any]] = []

    def dedupe_variants(kind: str) -> list[tuple[str, dict[str, bool]]]:
        return [
            (f"grow_propose_{kind}_plain", {}),
            (f"grow_propose_{kind}_dedupe_live", {"dedupe_live": True}),
            (f"grow_propose_{kind}_dedupe_step", {"dedupe_step": True}),
            (
                f"grow_propose_{kind}_dedupe_both",
                {"dedupe_live": True, "dedupe_step": True},
            ),
        ]

    for name, flags in dedupe_variants("per_unit"):
        units, edges = _growth_state()
        iu, ie = _units_json(units), _edges_json(edges)
        cands = ref.candidates_per_unit(
            units,
            proposals=2,
            network_seed=_GROW_SEED,
            step=_GROW_STEP,
            propose=rules.hash_propose_v1(n - 1),
        )
        cases.append(
            _grow_doc(
                name,
                "Per-unit proposals (P=2) under the site-rng contract; dedupe "
                "flags are opt-in and default off, so the plain variant may "
                "commit parallel and duplicate edges.",
                cands=cands,
                units=units,
                edges=edges,
                params={
                    "proposer": "per_unit",
                    "proposals_per_proposer": 2,
                    "selection": "top_k",
                    "max_new_per_level": 2,
                    "max_level_gap": 2,
                    **flags,
                },
                rules_used={
                    "propose": "hash_propose_v1",
                    "init": "grow_init_v1",
                },
                initial_units=iu,
                initial_edges=ie,
            )
        )

    for name, flags in dedupe_variants("per_conn"):
        units, edges = _growth_state()
        iu, ie = _units_json(units), _edges_json(edges)
        cands = ref.candidates_per_connection(
            units,
            edges,
            proposals=2,
            network_seed=_GROW_SEED,
            step=_GROW_STEP,
            propose=rules.conn_propose_v1(n - 1),
        )
        cases.append(
            _grow_doc(
                name,
                "Per-connection proposals: the proposer key hashes (src, dst, "
                "occurrence) so the parallel (2,4) edges draw distinct "
                "streams; the candidate index ranks proposers by ascending "
                "(src, dst, occurrence).",
                cands=cands,
                units=units,
                edges=edges,
                params={
                    "proposer": "per_connection",
                    "proposals_per_proposer": 2,
                    "selection": "top_k",
                    "max_new_per_level": 2,
                    "max_level_gap": 2,
                    **flags,
                },
                rules_used={
                    "propose": "conn_propose_v1",
                    "init": "grow_init_v1",
                },
                initial_units=iu,
                initial_edges=ie,
            )
        )

    for name, flags in dedupe_variants("global"):
        units, edges = _growth_state()
        iu, ie = _units_json(units), _edges_json(edges)
        cands = ref.candidates_global(
            proposals=6,
            network_seed=_GROW_SEED,
            step=_GROW_STEP,
            propose=rules.global_propose_v1(n - 1),
        )
        cases.append(
            _grow_doc(
                name,
                "Global proposals: one site (key 0), P=6, index = j.",
                cands=cands,
                units=units,
                edges=edges,
                params={
                    "proposer": "global",
                    "proposals_per_proposer": 6,
                    "selection": "top_k",
                    "max_new_per_level": 2,
                    "max_level_gap": 2,
                    **flags,
                },
                rules_used={
                    "propose": "global_propose_v1",
                    "init": "grow_init_v1",
                },
                initial_units=iu,
                initial_edges=ie,
            )
        )

    # Isolated unit: proposes per-unit but no connection proposes for it.
    units, edges = _growth_state(isolated_unit=True)
    iu, ie = _units_json(units), _edges_json(edges)
    cands = ref.candidates_per_connection(
        units,
        edges,
        proposals=2,
        network_seed=_GROW_SEED,
        step=_GROW_STEP,
        propose=rules.conn_propose_v1(n),
    )
    cases.append(
        _grow_doc(
            "grow_per_conn_isolated_unit_no_growth",
            "A unit with no incident live connection generates no "
            "per-connection candidates: unit 6 appears in no proposal site.",
            cands=cands,
            units=units,
            edges=edges,
            params={
                "proposer": "per_connection",
                "proposals_per_proposer": 2,
                "selection": "top_k",
                "max_new_per_level": 2,
                "max_level_gap": 2,
            },
            rules_used={"propose": "conn_propose_v1", "init": "grow_init_v1"},
            initial_units=iu,
            initial_edges=ie,
        )
    )

    # Scored strategies.
    for name, kind in (
        ("grow_score_exhaustive", "exhaustive"),
        ("grow_score_shortlist", "shortlist"),
        ("grow_score_predicate", "predicate"),
    ):
        units, edges = _growth_state()
        iu, ie = _units_json(units), _edges_json(edges)
        params: dict[str, Any]
        if kind == "exhaustive":
            cands = ref.candidates_exhaustive(
                units, capacity=_GROW_CAPACITY, score=rules.grid_score_v1
            )
            params = {"candidates": "exhaustive"}
            rl = {"score": "grid_score_v1", "init": "grow_init_v1"}
            comment = (
                "Exhaustive scored growth: every live ordered pair, index = "
                "src * capacity + dst; grid_score_v1 has deliberate ties."
            )
        elif kind == "shortlist":
            cands = ref.candidates_shortlist(
                units,
                shortlist_size=3,
                importance=rules.importance_v1,
                score=rules.grid_score_v1,
            )
            params = {"candidates": "shortlist", "shortlist_size": 3}
            rl = {
                "score": "grid_score_v1",
                "importance": "importance_v1",
                "init": "grow_init_v1",
            }
            comment = (
                "Shortlist: the M x M grid (M=3) over the importance-ranked "
                "units, row-major candidate index, importance ties broken by "
                "ascending unit id."
            )
        else:
            cands = ref.candidates_exhaustive(
                units, capacity=_GROW_CAPACITY, score=rules.predicate_score_v1
            )
            params = {
                "candidates": "exhaustive",
                "selection": "all",
                "dedupe_step": True,
            }
            rl = {"score": "predicate_score_v1", "init": "grow_init_v1"}
            comment = (
                "The predicate adapter maps should_add to scores 0 / -inf and "
                "fixes selection = all with dedupe_step = true."
            )
        base = {
            "selection": "top_k",
            "max_new_per_level": 2,
            "max_level_gap": 1,
        }
        base.update(params)
        cases.append(
            _grow_doc(
                name,
                comment,
                cands=cands,
                units=units,
                edges=edges,
                params=base,
                rules_used=rl,
                initial_units=iu,
                initial_edges=ie,
            )
        )

    # Per-level shortlist: one M x M grid per source level. M = 2 of the 3
    # units on every level, so both the source and the destination rankings
    # decide the grid, and it differs from the single global grid's.
    for name, direction in (
        ("grow_score_shortlist_per_level_any", "any"),
        ("grow_score_shortlist_per_level_deeper", "deeper"),
    ):
        units, edges = _per_level_state()
        iu, ie = _units_json(units), _edges_json(edges)
        cands = ref.candidates_shortlist_per_level(
            units,
            shortlist_size=2,
            max_level_gap=1,
            direction=direction,
            importance=rules.importance_v1,
            score=rules.grid_score_v1,
        )
        cases.append(
            _grow_doc(
                name,
                "Per-level shortlist (M=2): per source level, ascending, the "
                "sources are the level's top-M live units by importance and "
                "the destinations the top-M live units inside that level's "
                "validity window (level gap and direction), by importance; "
                "importance ties break by ascending unit id; index = "
                "level_rank * M * M + row-major position in the level's grid.",
                cands=cands,
                units=units,
                edges=edges,
                params={
                    "candidates": "shortlist_per_level",
                    "shortlist_size": 2,
                    "selection": "top_k",
                    "max_new_per_level": 2,
                    "max_level_gap": 1,
                    "direction": direction,
                },
                rules_used={
                    "score": "grid_score_v1",
                    "importance": "importance_v1",
                    "init": "grow_init_v1",
                },
                initial_units=iu,
                initial_edges=ie,
            )
        )

    # Selection stages on the exhaustive grid.
    selection_cases = (
        (
            "grow_select_topk_ties",
            {"selection": "top_k", "max_new_per_level": 3},
            "Equal scores fall back to (src, dst, index) ascending.",
        ),
        (
            "grow_select_threshold",
            {"selection": "threshold", "threshold": 0.25, "max_new_per_level": 3},
            "threshold keeps scores >= threshold(g), at most max_new_per_level.",
        ),
        (
            "grow_select_all",
            {"selection": "all"},
            "selection = all commits every finite candidate (capacity allowing).",
        ),
        (
            "grow_select_max_new_per_step",
            {"selection": "top_k", "max_new_per_level": 3, "max_new_per_step": 2},
            "max_new_per_step applies across levels, level ascending.",
        ),
        (
            "grow_select_overflow",
            {"selection": "all", "free_slots": 2},
            "Selected candidates beyond free capacity are dropped in order "
            "and raise conn_overflow (single claim domain).",
        ),
    )
    for name, extra, comment in selection_cases:
        units, edges = _growth_state()
        iu, ie = _units_json(units), _edges_json(edges)
        cands = ref.candidates_exhaustive(
            units, capacity=_GROW_CAPACITY, score=rules.grid_score_v1
        )
        free = extra.pop("free_slots", 8)
        params = {"candidates": "exhaustive", "max_level_gap": 2, **extra}
        cases.append(
            _grow_doc(
                name,
                comment,
                cands=cands,
                units=units,
                edges=edges,
                params=params,
                rules_used={"score": "grid_score_v1", "init": "grow_init_v1"},
                free_slots=free,
                initial_units=iu,
                initial_edges=ie,
            )
        )

    # Validity window sweep.
    for gap in (0, 1, 2):
        for direction in ("any", "deeper", "same_or_deeper"):
            units, edges = _growth_state()
            iu, ie = _units_json(units), _edges_json(edges)
            cands = ref.candidates_exhaustive(
                units, capacity=_GROW_CAPACITY, score=rules.grid_score_v1
            )
            cases.append(
                _grow_doc(
                    f"grow_window_gap{gap}_{direction}",
                    "Validity window: |level(dst) - level(src)| <= gap plus "
                    "the direction constraint; failures score -inf.",
                    cands=cands,
                    units=units,
                    edges=edges,
                    params={
                        "candidates": "exhaustive",
                        "selection": "all",
                        "max_level_gap": gap,
                        "direction": direction,
                    },
                    rules_used={"score": "grid_score_v1", "init": "grow_init_v1"},
                    initial_units=iu,
                    initial_edges=ie,
                )
            )
    for loops in (False, True):
        units, edges = _growth_state()
        iu, ie = _units_json(units), _edges_json(edges)
        cands = ref.candidates_exhaustive(
            units, capacity=_GROW_CAPACITY, score=rules.grid_score_v1
        )
        cases.append(
            _grow_doc(
                f"grow_self_loops_{'on' if loops else 'off'}",
                "src == dst is vetoed unless allow_self_loops.",
                cands=cands,
                units=units,
                edges=edges,
                params={
                    "candidates": "exhaustive",
                    "selection": "all",
                    "max_level_gap": 0,
                    "allow_self_loops": loops,
                },
                rules_used={"score": "grid_score_v1", "init": "grow_init_v1"},
                initial_units=iu,
                initial_edges=ie,
            )
        )

    # Triggers: a false trigger makes the phase a no-op.
    trigger_cases = (
        ("grow_trigger_every3_fires", {"kind": "every", "n": 3, "step": 3}, True),
        ("grow_trigger_every3_skips", {"kind": "every", "n": 3, "step": 4}, False),
        (
            "grow_trigger_on_units_added_fires",
            {"kind": "on_units_added", "units_added_this_step": 2},
            True,
        ),
        (
            "grow_trigger_on_units_added_skips",
            {"kind": "on_units_added", "units_added_this_step": 0},
            False,
        ),
        ("grow_trigger_when_fires", {"kind": "when", "value": True}, True),
        ("grow_trigger_when_skips", {"kind": "when", "value": False}, False),
    )
    for name, trig, fires in trigger_cases:
        units, edges = _growth_state()
        iu, ie = _units_json(units), _edges_json(edges)
        step = int(trig.get("step", _GROW_STEP))
        if fires:
            cands = ref.candidates_exhaustive(
                units, capacity=_GROW_CAPACITY, score=rules.grid_score_v1
            )
            doc = _grow_doc(
                name,
                "every(n) fires on steps where step % n == 0; on_units_added "
                "fires iff this step's add-unit phase created units; when(g) "
                "evaluates the user predicate on globals.",
                cands=cands,
                units=units,
                edges=edges,
                params={
                    "candidates": "exhaustive",
                    "selection": "top_k",
                    "max_new_per_level": 2,
                    "max_level_gap": 2,
                    "step": step,
                },
                rules_used={"score": "grid_score_v1", "init": "grow_init_v1"},
                initial_units=iu,
                initial_edges=ie,
            )
        else:
            doc = _doc(
                name,
                "growth_v2",
                "every(n) fires on steps where step % n == 0; on_units_added "
                "fires iff this step's add-unit phase created units; when(g) "
                "evaluates the user predicate on globals. A false trigger "
                "makes the phase a no-op.",
                rules={"score": "grid_score_v1", "init": "grow_init_v1"},
                params={
                    "capacity": _GROW_CAPACITY,
                    "free_slots": 8,
                    "network_seed": _GROW_SEED,
                    "step": step,
                    "candidates": "exhaustive",
                    "selection": "top_k",
                    "max_new_per_level": 2,
                    "max_level_gap": 2,
                },
                initial_units=iu,
                initial_edges=ie,
                expect={
                    "committed": [],
                    "grown": 0,
                    "conn_overflow": False,
                    "needs_resort": False,
                },
            )
        doc["trigger"] = trig
        cases.append(doc)

    # Batched: growth reads the batch-mean state (topological only).
    units, edges = _growth_state()
    batch = [
        [0.75, -0.5, 0.25, 0.5, -0.75, 0.5],
        [0.25, 0.0, 0.0, 1.0, -0.25, 0.0],
    ]
    mean = [
        float(np.float32((np.float32(a) + np.float32(b)) / np.float32(2)))
        for a, b in zip(*batch, strict=True)
    ]
    for u, m in zip(units, mean, strict=True):
        u["fields"]["activation"] = m
    iu, ie = _units_json(units), _edges_json(edges)
    cands = ref.candidates_exhaustive(
        units, capacity=_GROW_CAPACITY, score=rules.score_act_v1(units)
    )
    doc = _grow_doc(
        "grow_batched_mean_state",
        "Under batching, growth runs once per batched step on the batch-mean "
        "unit state (topological mode only); score_act_v1 reads the mean "
        "activations, making the dependence observable.",
        cands=cands,
        units=units,
        edges=edges,
        params={
            "candidates": "exhaustive",
            "selection": "top_k",
            "max_new_per_level": 2,
            "max_level_gap": 1,
        },
        rules_used={"score": "score_act_v1", "init": "grow_init_v1"},
        initial_units=iu,
        initial_edges=ie,
    )
    doc["batch_activations"] = batch
    cases.append(doc)

    return cases


def _claim_select(
    params: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """The shared growth net's exhaustive grid_score_v1 selection."""
    units, edges = _growth_state()
    cands = ref.candidates_exhaustive(
        units, capacity=_GROW_CAPACITY, score=rules.grid_score_v1
    )
    selected = ref.select_growth(units, edges, cands, capacity=_GROW_CAPACITY, **params)
    return units, edges, selected


def _claim_cases() -> list[dict[str, Any]]:
    """The per-model slot claim: per-level free accounting and regrow."""
    cases: list[dict[str, Any]] = []
    init_rules = {"score": "grid_score_v1", "init": "grow_init_v1"}
    base = {
        "capacity": _GROW_CAPACITY,
        "network_seed": _GROW_SEED,
        "step": _GROW_STEP,
        "candidates": "exhaustive",
    }

    # Topological: strict per-level buckets, then a regrow and a retry.
    select = {
        "max_level_gap": 1,
        "selection": "top_k",
        "max_new_per_level": 3,
        "dedupe_live": True,
    }
    free = {0: 1, 1: 6, 2: 3}
    units, edges, selected = _claim_select(select)
    iu, ie = _units_json(units), _edges_json(edges)
    first = ref.commit_growth(
        units,
        edges,
        selected,
        free_per_level=free,
        init=rules.grow_init_v1,
        field_defaults={"weight": 0.0},
    )
    assert first["conn_overflow"] and first["regrow_levels"] == [0, 2]
    # The retry: the next step's growth over the attempt's output edges; a
    # regrown bucket holds 8 free slots, the others what the attempt left.
    retry_free = {
        lvl: 8
        if lvl in first["regrow_levels"]
        else f - sum(units[c["src"]]["level"] == lvl for c in first["committed"])
        for lvl, f in free.items()
    }
    after_edges = copy.deepcopy(edges)
    cands = ref.candidates_exhaustive(
        units, capacity=_GROW_CAPACITY, score=rules.grid_score_v1
    )
    retry_selected = ref.select_growth(
        units, after_edges, cands, capacity=_GROW_CAPACITY, **select
    )
    retry = ref.commit_growth(
        units,
        after_edges,
        retry_selected,
        free_per_level=retry_free,
        init=rules.grow_init_v1,
        field_defaults={"weight": 0.0},
    )
    assert not retry["conn_overflow"]
    doc = _doc(
        "grow_claim_topological_regrow",
        "growth_v2",
        "Topological claim: each source level claims only its own bucket's "
        "free slots, so level 0 overflows while level 1 keeps spare slots. "
        "The host loop then grows every full bucket (regrow_levels) and "
        "re-runs the step; the retry block is that step's growth, at step + "
        "1, over the attempt's output edges with the regrown free counts.",
        rules=init_rules,
        params={
            **base,
            "model": "topological",
            "free_per_level": {str(k): v for k, v in free.items()},
            **select,
        },
        initial_units=iu,
        initial_edges=ie,
        expect=first,
    )
    doc["retry"] = {
        "step": _GROW_STEP + 1,
        "free_per_level": {str(k): v for k, v in retry_free.items()},
        "expect": retry,
    }
    cases.append(doc)

    # Pipeline: a level's own dead slots first, then the shared tail.
    select = {"max_level_gap": 1, "direction": "deeper", "selection": "all"}
    pipeline_cases = (
        (
            "grow_claim_pipeline_spill",
            [(0, 2), (1, 3), (2, 5), (4, 5)],
            6,
            "Pipeline claim: each level first takes the dead slots whose "
            "former occupant it sourced, then spills to the never-used tail, "
            "levels ascending. Level 2's dead slot stays free: no level "
            "borrows another's.",
        ),
        (
            "grow_claim_pipeline_overflow",
            [(0, 2), (2, 5)],
            2,
            "Pipeline claim overflow: each level's best candidate takes its "
            "own dead slot; the spills reach the tail levels ascending, not "
            "by score, so level 0's exhaust it ahead of level 1's "
            "better-scored ones, which are dropped with conn_overflow.",
        ),
    )
    for name, dead_pairs, tail, comment in pipeline_cases:
        units, edges, selected = _claim_select(select)
        iu, ie = _units_json(units), _edges_json(edges)
        dead_per_level: dict[int, int] = {}
        for src, _ in dead_pairs:
            lvl = units[src]["level"]
            dead_per_level[lvl] = dead_per_level.get(lvl, 0) + 1
        outcome = ref.commit_growth(
            units,
            edges,
            selected,
            dead_per_level=dead_per_level,
            tail=tail,
            init=rules.grow_init_v1,
            field_defaults={"weight": 0.0},
        )
        cases.append(
            _doc(
                name,
                "growth_v2",
                comment,
                rules=init_rules,
                params={
                    **base,
                    "model": "pipeline",
                    "dead_per_level": {
                        str(k): v for k, v in sorted(dead_per_level.items())
                    },
                    "tail": tail,
                    **select,
                },
                initial_units=iu,
                initial_edges=ie,
                initial_dead_edges=[
                    {"src": src, "dst": dst} for src, dst in dead_pairs
                ],
                expect=outcome,
            )
        )
    return cases


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_all() -> dict[str, dict[str, Any]]:
    """Every registry golden, keyed by filename."""
    docs = [_passes_topological(), _passes_pipeline()]
    docs.extend(_loss_cases())
    docs.extend(_unit_cases())
    docs.extend(_grow_cases())
    docs.extend(_claim_cases())
    return {f"{d['name']}.json": d for d in docs}


def _render(doc: dict[str, Any]) -> str:
    return json.dumps(doc, indent=1, sort_keys=True) + "\n"


def main() -> int:
    """Emit or check the registry goldens.

    Returns:
        Process exit code: 0 on success, 1 on a --check mismatch.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify committed goldens match regeneration byte-for-byte",
    )
    args = parser.parse_args()

    docs = build_all()
    destinations = [_PX_GOLDEN]
    cx = _cx_golden_dir()
    if cx is not None:
        destinations.append(cx)

    if args.check:
        failed = []
        for dest in destinations:
            for name, doc in docs.items():
                path = dest / name
                if not path.is_file() or path.read_text() != _render(doc):
                    failed.append(str(path))
        if failed:
            print(f"STALE ({len(failed)}): " + ", ".join(sorted(failed)))
            print("regenerate with: uv run python scripts/parity/emit.py")
            return 1
        checked = len(docs) * len(destinations)
        print(f"OK: {checked} registry golden files match their generator")
        return 0

    for dest in destinations:
        dest.mkdir(parents=True, exist_ok=True)
        for name, doc in docs.items():
            (dest / name).write_text(_render(doc))
        print(f"wrote {len(docs)} goldens to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
