"""Consume the registry goldens (`tests/golden/*.json`, schema registry_v1).

The goldens are emitted by `scripts/parity/emit.py` from the pure-NumPy
reference in `scripts/parity/reference.py`; plastax-cpp consumes the same
files. Each golden names the feature set it `requires`:

- ``passes_v1`` runs here today, with exact float equality (every value in
  those goldens is a dyadic fraction, so float32 arithmetic on them is exact).
- ``loss_v1`` runs the shipped `SoftmaxCrossEntropyLoss` through a real step:
  the gradient seed on every output and the returned loss, exactly.
- ``growth_v2`` is enforced in full: the propose cases and the score cases
  (exhaustive/shortlist/per-level shortlist/predicate scoring, every
  selection mode, the validity window, the triggers, and growth on the
  batch-mean state).
- ``unit_lifecycle_v1``: the unit-update cases (those whose only rule is
  ``update_unit``) are enforced. Unit pruning and addition are not implemented
  yet; their goldens are skipped loudly below, one visible skip per file,
  until the features land.

The reference's Philox core is additionally pinned, bit for bit, to the same
`rng_philox32.json` golden the shipped `plastax.rng` is pinned to, so the
spec's randomness and the library's randomness cannot drift apart.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import plastax as px

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts/parity"))
import reference as ref  # noqa: E402

from _plastax_cpp import plastax_cpp_dir  # noqa: E402

_GOLDEN_DIR = pathlib.Path(__file__).resolve().parent / "golden"
_IMPLEMENTED = {"passes_v1", "loss_v1"}
_SPEC_ONLY = {
    "unit_lifecycle_v1": "unit pruning and addition are not implemented",
    "growth_v2": "no consumer covers this growth_v2 case",
}
# growth_v2 propose cases (enforced by test_grow_propose_golden).
_ENFORCED_GROWTH = {
    f"grow_propose_{kind}_{var}"
    for kind in ("per_unit", "per_conn", "global")
    for var in ("plain", "dedupe_live", "dedupe_step", "dedupe_both")
} | {"grow_per_conn_isolated_unit_no_growth"}
# growth_v2 score cases (enforced by test_grow_score_golden): every one whose
# rules name a `score`, discovered so a newly emitted case cannot slip past.
_ENFORCED_SCORE = {
    path.stem
    for path in sorted(_GOLDEN_DIR.glob("grow_*.json"))
    if json.loads(path.read_text()).get("requires") == "growth_v2"
    and "score" in json.loads(path.read_text())["rules"]
}

# unit_lifecycle_v1 update cases (enforced by test_unit_update_golden): every
# one whose only rule is update_unit, discovered like the score cases.
_ENFORCED_UNIT = {
    path.stem
    for path in sorted(_GOLDEN_DIR.glob("unit_*.json"))
    if json.loads(path.read_text()).get("requires") == "unit_lifecycle_v1"
    and set(json.loads(path.read_text())["rules"]) == {"update_unit"}
}


# loss_v1 cases (enforced by test_loss_golden), discovered by tag.
_ENFORCED_LOSS = {
    path.stem
    for path in sorted(_GOLDEN_DIR.glob("loss_*.json"))
    if json.loads(path.read_text()).get("requires") == "loss_v1"
}


def _registry_goldens() -> list[pathlib.Path]:
    return sorted(
        p
        for p in _GOLDEN_DIR.glob("*.json")
        if json.loads(p.read_text()).get("schema") == "registry_v1"
    )


def _load(name: str) -> dict[str, Any]:
    return json.loads((_GOLDEN_DIR / name).read_text())


# ---------------------------------------------------------------------------
# The reference RNG is pinned to the same golden as the shipped rng.
# ---------------------------------------------------------------------------

_CPP_PHILOX = plastax_cpp_dir() / "tests" / "golden" / "rng_philox32.json"


@pytest.mark.skipif(
    not _CPP_PHILOX.is_file(),
    reason=f"no plastax-cpp Philox golden at {_CPP_PHILOX}; set PLASTAX_CPP_DIR",
)
def test_reference_philox_is_bit_exact() -> None:
    """Every golden (seed, counter) reproduces the exact word."""
    golden = json.loads(_CPP_PHILOX.read_text())
    assert len(golden["samples"]) > 100
    for sample in golden["samples"]:
        word = ref.philox32(sample["seed"], sample["counter"])
        assert format(word, "08x") == sample["word"], (
            f"philox32({sample['seed']}, {sample['counter']})"
        )


# ---------------------------------------------------------------------------
# passes_v1: the dyadic ReLU MLP, exact equality.
# ---------------------------------------------------------------------------

GRAD_PRE_ACT = px.FieldSpec.float32("grad_pre_act")
LOSS_GRAD = px.FieldSpec.float32("loss_grad")


class _WeightedSumMap:
    combine = px.monoid.sum_

    def map(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> jax.Array:
        """weight * activation[src] for one edge."""
        del dst, g
        return c[px.WEIGHT, cid] * u[px.ACTIVATION, src]


class ReluForward(_WeightedSumMap, px.ForwardPass):
    """apply = max(acc, 0); mirrors golden rule relu_mlp_v1."""

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jax.Array
    ) -> px.UnitWrite:
        """Write the rectified accumulated input."""
        del u, i, g
        return px.UnitWrite.of((px.ACTIVATION, jnp.maximum(acc, jnp.float32(0.0))))


class ReluBackward(px.BackwardPass):
    """grad_pre_act = (acc + loss_grad) * [activation > 0]."""

    combine = px.monoid.sum_

    def map(
        self,
        u: px.UnitView,
        src: px.UnitIdx,
        dst: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> jax.Array:
        """weight * grad_pre_act[dst] for one outgoing edge."""
        del src, g
        return c[px.WEIGHT, cid] * u[GRAD_PRE_ACT, dst]

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jax.Array
    ) -> px.UnitWrite:
        """Gate the accumulated gradient by the ReLU derivative."""
        del g
        gate = jnp.where(u[px.ACTIVATION, i] > 0, jnp.float32(1.0), jnp.float32(0.0))
        return px.UnitWrite.of((GRAD_PRE_ACT, (acc + u[LOSS_GRAD, i]) * gate))


class MSELoss(px.Loss):
    """L = 0.5*sum((pred - target)^2); seeds dL/dpred into loss_grad."""

    seed_field = LOSS_GRAD

    def calculate_loss(
        self, u: px.UnitView, outputs: jax.Array, targets: jax.Array, g: None
    ) -> tuple[jax.Array, jax.Array]:
        """Return the loss and the gradient seed of every output."""
        del g
        diff = u.gather(px.ACTIVATION, outputs) - targets
        return jnp.sum(jnp.float32(0.5) * diff * diff), diff


class _ReluMlpNet(px.Network[None]):
    forward_pass = ReluForward()
    backward_pass = ReluBackward()
    loss = MSELoss()
    extra_unit_fields = (GRAD_PRE_ACT, LOSS_GRAD)
    propagation = px.Propagation.TOPOLOGICAL


class _ReluPipelineNet(px.Network[None]):
    forward_pass = ReluForward()
    propagation = px.Propagation.PIPELINE


def _dyadic_weight(src: int, dst: int) -> float:
    """w(src, dst) = (((3*src + 5*dst) mod 16) - 8) / 8 (golden_rules v1)."""
    return float(np.float32((((3 * src + 5 * dst) % 16) - 8) / 8.0))


def _build(net: type[px.Network[None]], doc: dict[str, Any]) -> tuple[Any, Any]:
    spec = doc["network"]
    blocks = [px.topology.input_units(spec["input_dim"])]
    start_src = 0
    n_src = spec["input_dim"]
    for width in spec["layers"]:
        start_dst = start_src + n_src
        w = np.zeros((n_src, width), dtype=np.float32)
        for s in range(n_src):
            for d in range(width):
                w[s, d] = _dyadic_weight(start_src + s, start_dst + d)

        def init(key: Any, shape: tuple[int, ...], _w: np.ndarray = w) -> Any:
            del key
            assert shape == _w.shape
            return jnp.asarray(_w)

        blocks.append(px.topology.dense(n_src, width, init=init))
        start_src, n_src = start_dst, width
    return px.NetworkBuilder.from_topology(
        net, px.topology.sequential(*blocks), jax.random.PRNGKey(0), globals_=None
    )


def test_passes_relu_topological_matches_golden_exactly() -> None:
    """Forward, MSE loss and backward agree with the reference bit-for-bit."""
    doc = _load("passes_relu_topological.json")
    static, state = _build(_ReluMlpNet, doc)
    step = px.make_step(_ReluMlpNet, static)
    for i, s in enumerate(doc["steps"]):
        result = step(
            state,
            px.StepInputs(
                inputs=jnp.asarray(s["inputs"], jnp.float32),
                targets=jnp.asarray(s["targets"], jnp.float32),
            ),
        )
        state = result.state
        acts = np.asarray(state.units[px.ACTIVATION.name])
        grads = np.asarray(state.units[GRAD_PRE_ACT.name])
        assert acts.tolist() == s["expect"]["activations"], f"step {i} activations"
        assert grads.tolist() == s["expect"]["grad_pre_act"], f"step {i} grads"
        assert float(result.loss) == s["expect"]["loss"], f"step {i} loss"


def test_passes_relu_pipeline_matches_golden_exactly() -> None:
    """Pipeline forward: one connection crossed per step, exactly."""
    doc = _load("passes_relu_pipeline.json")
    static, state = _build(_ReluPipelineNet, doc)
    step = px.make_step(_ReluPipelineNet, static)
    for i, s in enumerate(doc["steps"]):
        result = step(
            state,
            px.StepInputs(inputs=jnp.asarray(s["inputs"], jnp.float32), targets=None),
        )
        state = result.state
        acts = np.asarray(state.units[px.ACTIVATION.name])
        assert acts.tolist() == s["expect"]["activations"], f"step {i} activations"


# ---------------------------------------------------------------------------
# loss_v1: the whole-output loss contract, through softmax cross-entropy.
# ---------------------------------------------------------------------------


class LinearForward(_WeightedSumMap, px.ForwardPass):
    """apply = acc; golden rule linear_v1."""

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jax.Array
    ) -> px.UnitWrite:
        """Write the accumulated input unchanged."""
        del u, i, g
        return px.UnitWrite.of((px.ACTIVATION, acc))


class _SoftmaxNet(px.Network[None]):
    forward_pass = LinearForward()
    loss = px.SoftmaxCrossEntropyLoss(LOSS_GRAD)
    extra_unit_fields = (LOSS_GRAD,)
    propagation = px.Propagation.TOPOLOGICAL


@pytest.mark.parametrize("name", sorted(_ENFORCED_LOSS))
def test_loss_golden(name: str) -> None:
    """The shipped softmax cross-entropy reproduces seed and loss exactly."""
    doc = _load(f"{name}.json")
    assert doc["rules"] == {
        "loss": "softmax_ce_v1",
        "forward": "linear_v1",
        "weights": "identity_weight_v1",
    }
    n_in = doc["network"]["input_dim"]
    (n_out,) = doc["network"]["layers"]
    edges = doc["initial_edges"]
    output_ids = list(range(n_in, n_in + n_out))
    static, state = px.NetworkBuilder.from_edges(
        _SoftmaxNet,
        n_in + n_out,
        np.asarray([e["src"] for e in edges], dtype=np.int32),
        np.asarray([e["dst"] for e in edges], dtype=np.int32),
        weights=np.asarray([e["fields"]["weight"] for e in edges], dtype=np.float32),
        input_ids=list(range(n_in)),
        output_ids=output_ids,
        globals_=None,
    )
    step = px.make_step(_SoftmaxNet, static)
    for i, s in enumerate(doc["steps"]):
        result = step(
            state,
            px.StepInputs(
                inputs=jnp.asarray(s["inputs"], jnp.float32),
                targets=jnp.asarray(s["targets"], jnp.float32),
            ),
        )
        state = result.state
        acts = np.asarray(state.units[px.ACTIVATION.name])[output_ids]
        seed = np.asarray(state.units[LOSS_GRAD.name])[output_ids]
        assert acts.tolist() == s["expect"]["activations"], f"step {i} activations"
        assert seed.tolist() == s["expect"]["seed"], f"step {i} seed"
        assert float(result.loss) == s["expect"]["loss"], f"step {i} loss"


# ---------------------------------------------------------------------------
# Spec-only goldens: loud per-file skips until the features land.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _registry_goldens(), ids=lambda p: p.stem)
def test_registry_golden_is_consumed_or_knowingly_skipped(path: pathlib.Path) -> None:
    """Every registry golden is either consumed above or skipped by name."""
    doc = json.loads(path.read_text())
    requires = doc["requires"]
    if requires in _IMPLEMENTED:
        assert (
            doc["name"]
            in {
                "passes_relu_topological",
                "passes_relu_pipeline",
            }
            | _ENFORCED_LOSS
        ), f"{doc['name']} claims {requires} but no consumer covers it"
        return
    assert requires in _SPEC_ONLY, f"unknown requires tag {requires!r} in {path.name}"
    if doc["name"] in _ENFORCED_GROWTH | _ENFORCED_SCORE | _ENFORCED_UNIT:
        return  # consumed by the test_grow_* / test_unit_update_golden tests
    pytest.skip(f"{doc['name']}: {_SPEC_ONLY[requires]} ({requires})")


# ---------------------------------------------------------------------------
# growth_v2, propose cases: the real pipeline against the reference, exactly.
# ---------------------------------------------------------------------------


def _dyadic_weight_arr(src: jax.Array, dst: jax.Array) -> jax.Array:
    """grow_init_v1's weight, vectorized: (((3*src + 5*dst) % 16) - 8) / 8."""
    return (((3 * src + 5 * dst) % 16) - 8).astype(jnp.float32) / jnp.float32(8.0)


def _registry_score(rng: Any) -> jax.Array:
    """The registry propose rules' score: floor(uniform * 256) / 256."""
    return jnp.floor(rng.uniform() * jnp.float32(256.0)) / jnp.float32(256.0)


def _make_propose_rule(doc: dict[str, Any], n_units: int) -> px.ProposeAddConn[None]:
    """The golden's registry propose rule, on the shipped rng."""
    params = doc["params"]
    assert doc["rules"]["propose"] in {
        "hash_propose_v1",
        "conn_propose_v1",
        "global_propose_v1",
    }
    assert doc["rules"]["init"] == "grow_init_v1"
    kind = params["proposer"]

    class _Rule(px.ProposeAddConn[None]):
        proposer = kind
        proposals_per_proposer = int(params["proposals_per_proposer"])
        max_new_per_level = int(params["max_new_per_level"])
        max_level_gap = int(params["max_level_gap"])
        dedupe_live = bool(params.get("dedupe_live", False))
        dedupe_step = bool(params.get("dedupe_step", False))

        def propose(  # type: ignore[override]
            self, *args: Any
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            if kind == "per_unit":
                u, i, j, g, rng = args
                del u, j, g
                src = jnp.asarray(i, jnp.int32)
            elif kind == "per_connection":
                u, c, cid, j, g, rng = args
                del u, j, g
                src = jnp.asarray(c[px.FROM_ID, cid], jnp.int32)
            else:
                u, j, g, rng = args
                del u, j, g
                src = rng.uniform_int(n_units).astype(jnp.int32)
            dst = rng.uniform_int(n_units).astype(jnp.int32)
            return src, dst, _registry_score(rng)

        def init(
            self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
        ) -> px.ConnWrite:
            del u, g
            return px.ConnWrite.of((px.WEIGHT, _dyadic_weight_arr(src, dst)))

    return _Rule()


def _live_pairs(state: Any) -> list[tuple[int, int, float]]:
    out = []
    for bucket in state.conns:
        dead = np.asarray(bucket[px.DEAD.name])
        srcs = np.asarray(bucket[px.FROM_ID.name])
        dsts = np.asarray(bucket[px.TO_ID.name])
        ws = np.asarray(bucket[px.WEIGHT.name])
        for i in range(len(dead)):
            if not dead[i]:
                out.append((int(srcs[i]), int(dsts[i]), float(ws[i])))
    return sorted(out)


@pytest.mark.parametrize("name", sorted(_ENFORCED_GROWTH))
def test_grow_propose_golden(name: str) -> None:
    """The propose pipeline reproduces the reference commit set exactly."""
    import dataclasses

    from plastax.phases import build_add_conn_phase

    doc = _load(f"{name}.json")
    params = doc["params"]
    units = doc["initial_units"]
    n_units = len(units)
    rule = _make_propose_rule(doc, n_units)

    class _Net(px.Network[None]):
        forward_pass = ReluForward()
        add_conn = rule
        seed = int(params["network_seed"])
        propagation = px.Propagation.TOPOLOGICAL

    edges = doc["initial_edges"]
    static, state = px.NetworkBuilder.from_edges(
        _Net,
        n_units,
        np.asarray([e["src"] for e in edges], dtype=np.int32),
        np.asarray([e["dst"] for e in edges], dtype=np.int32),
        weights=np.asarray([e["fields"]["weight"] for e in edges], dtype=np.float32),
        input_ids=[u["id"] for u in units if u["is_input"]],
        output_ids=[u["id"] for u in units if u["is_output"]],
        globals_=None,
        capacity_headroom=4.0,
    )
    # The builder derives levels by Kahn; the golden's declared levels must
    # agree, or the window/bucket routing would diverge from the reference.
    # Exception: a unit with no incident edges always derives level 0 in px,
    # whatever the golden declares -- tolerated only because such a unit
    # proposes nothing per-connection and the committed-set assertion below
    # still catches any window effect of the differing level.
    touched = {e["src"] for e in edges} | {e["dst"] for e in edges}
    got_levels = np.asarray(state.units[px.LEVEL.name]).tolist()
    for u in units:
        if u["id"] in touched:
            assert got_levels[u["id"]] == u["level"], f"unit {u['id']} level"
    state = dataclasses.replace(state, step=jnp.int32(params["step"]))
    before = _live_pairs(state)
    phase = build_add_conn_phase(_Net, static)
    new_state, _ = phase(state, px.StepInputs(inputs=jnp.zeros((0,)), targets=None))

    after = _live_pairs(new_state)
    grown = sorted(after)
    for pair in before:
        grown.remove(pair)
    want = sorted(
        (
            int(c["src"]),
            int(c["dst"]),
            float(np.float32((((3 * c["src"] + 5 * c["dst"]) % 16) - 8) / 8.0)),
        )
        for c in doc["expect"]["committed"]
    )
    assert grown == want, f"{name}: committed edges diverge from the reference"
    assert len(grown) == doc["expect"]["grown"]
    assert bool(new_state.needs_resort) == doc["expect"]["needs_resort"]


# ---------------------------------------------------------------------------
# growth_v2, score cases: the real pipeline against the reference, exactly.
#
# Claim domains. The reference claims into ONE domain of `free_slots` slots
# (cx's single connection arena): the selection -- level-ascending, then the
# total order -- is committed as a prefix, and anything beyond raises the
# overflow flag. px claims per source-level bucket instead, so the harness
# maps the single domain onto px's own semantics rather than onto the
# expected answer:
#
# - run A (ample bucket capacity) yields px's full selection S;
# - run B caps the step at `free_slots` through `max_new_per_step`, which is
#   precisely the level-ascending, total-order prefix -- its commits must
#   equal the reference's exactly, and the reference's overflow flag must
#   equal |S| > free_slots;
# - run C runs px's real per-bucket claim with the single domain's slots
#   distributed the way its prefix spends them -- level L gets
#   clamp(free_slots - |S below L|, 0, |S at L|), derived from px's own
#   selection S, never from the expected commits -- and checks the claim's
#   own commits and overflow flag against the reference.
# ---------------------------------------------------------------------------


def _grid_score_v1(src: jax.Array, dst: jax.Array) -> jax.Array:
    """grid_score_v1: (((3*src + 5*dst) mod 17) - 8) / 8."""
    return (((3 * src + 5 * dst) % 17) - 8).astype(jnp.float32) / jnp.float32(8.0)


def _importance_v1(i: jax.Array) -> jax.Array:
    """importance_v1: ((7*i) mod 13) / 4."""
    return ((7 * i) % 13).astype(jnp.float32) / jnp.float32(4.0)


def _make_score_rule(
    doc: dict[str, Any], *, step_cap: int | None
) -> px.ScoreAddConn[dict[str, jax.Array]]:
    """The golden's registry score rule, every knob from its params."""
    params = doc["params"]
    rules = doc["rules"]
    assert rules["init"] == "grow_init_v1"
    score_rule = rules["score"]
    assert score_rule in {"grid_score_v1", "predicate_score_v1", "score_act_v1"}

    def init(
        u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: dict[str, jax.Array]
    ) -> px.ConnWrite:
        del u, g
        return px.ConnWrite.of((px.WEIGHT, _dyadic_weight_arr(src, dst)))

    knobs: dict[str, Any] = {
        "max_level_gap": int(params["max_level_gap"]),
        "selection": params["selection"],
        "direction": params.get("direction", "any"),
        "allow_self_loops": bool(params.get("allow_self_loops", False)),
        "dedupe_live": bool(params.get("dedupe_live", False)),
        "dedupe_step": bool(params.get("dedupe_step", False)),
    }
    if "max_new_per_level" in params:
        knobs["max_new_per_level"] = int(params["max_new_per_level"])
    caps = [c for c in (params.get("max_new_per_step"), step_cap) if c is not None]
    if caps:
        knobs["max_new_per_step"] = int(min(caps))
    trigger = doc.get("trigger")
    if trigger is not None:
        knobs["trigger"] = (
            ("every", int(trigger["n"]))
            if trigger["kind"] == "every"
            else trigger["kind"]
        )

    if score_rule == "predicate_score_v1":
        # The adapter itself is under test: True -> 0.0, False -> -inf,
        # selection = "all", dedupe_step = True -- the golden's params must
        # agree with the adapter's defaults rather than override them.
        assert knobs.pop("selection") == "all"
        assert knobs.pop("dedupe_step") is True

        def should_add(
            u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: dict[str, jax.Array]
        ) -> jax.Array:
            del u, g
            return (src + dst) % 2 == 0

        return px.predicate_add_conn(should_add, init, **knobs)

    class _Rule:
        if params["candidates"] in ("shortlist", "shortlist_per_level"):
            candidates = params["candidates"]
            shortlist_size = int(params["shortlist_size"])

        def score(
            self,
            u: px.UnitView,
            src: px.UnitIdx,
            dst: px.UnitIdx,
            g: dict[str, jax.Array],
        ) -> jax.Array:
            del g
            if score_rule == "score_act_v1":
                return u[px.ACTIVATION, src] + u[px.ACTIVATION, dst]
            return _grid_score_v1(src, dst)

        def importance(
            self, u: px.UnitView, i: px.UnitIdx, g: dict[str, jax.Array]
        ) -> jax.Array:
            del u, g
            return _importance_v1(i)

        def threshold(self, g: dict[str, jax.Array]) -> jax.Array:
            return g["threshold"]

        def when(self, g: dict[str, jax.Array]) -> jax.Array:
            return g["when"]

        def init(
            self,
            u: px.UnitView,
            src: px.UnitIdx,
            dst: px.UnitIdx,
            g: dict[str, jax.Array],
        ) -> px.ConnWrite:
            return init(u, src, dst, g)

    for key, value in knobs.items():
        setattr(_Rule, key, value)
    return _Rule()


def _score_net(
    doc: dict[str, Any], *, step_cap: int | None, free_at: dict[int, int]
) -> tuple[Any, Any, Any]:
    """Build the golden's network; `free_at` pins exact free-slot counts.

    Every bucket is built with ample capacity. For each `free_at[L] = f`,
    bucket L's surplus dead slots (all but the first f) are plugged with a
    live, zero-weight self-loop on a level-L unit, leaving exactly f free
    slots. The plug is inert to these goldens (none sets `dedupe_live`, and
    scores read only ids and activations) and is diffed out of the commits.
    """
    import dataclasses

    params = doc["params"]
    units = doc["initial_units"]
    n_units = len(units)
    rule = _make_score_rule(doc, step_cap=step_cap)

    class _Net(px.Network[dict[str, jax.Array]]):
        forward_pass = ReluForward()
        add_conn = rule
        seed = int(params["network_seed"])
        propagation = px.Propagation.TOPOLOGICAL

    edges = doc["initial_edges"]
    trigger = doc.get("trigger") or {}
    globals_ = {
        "threshold": jnp.float32(params.get("threshold", 0.0)),
        "when": jnp.bool_(trigger.get("value", True)),
    }
    static, state = px.NetworkBuilder.from_edges(
        _Net,
        n_units,
        np.asarray([e["src"] for e in edges], dtype=np.int32),
        np.asarray([e["dst"] for e in edges], dtype=np.int32),
        weights=np.asarray([e["fields"]["weight"] for e in edges], dtype=np.float32),
        input_ids=[u["id"] for u in units if u["is_input"]],
        output_ids=[u["id"] for u in units if u["is_output"]],
        globals_=globals_,
        capacity_headroom=4.0,
    )
    got_levels = np.asarray(state.units[px.LEVEL.name]).tolist()
    for u in units:
        assert got_levels[u["id"]] == u["level"], f"unit {u['id']} level"
    if free_at:
        conns = list(state.conns)
        for lvl, free in free_at.items():
            bucket = dict(conns[lvl])
            dead = np.flatnonzero(np.asarray(bucket[px.DEAD.name]))
            assert dead.size >= free, "bucket built too small to pin"
            surplus = dead[free:]
            plug = next(u["id"] for u in units if u["level"] == lvl)
            plug_value = {
                px.DEAD.name: False,
                px.FROM_ID.name: plug,
                px.TO_ID.name: plug,
                px.WEIGHT.name: 0.0,
            }
            for name, value in plug_value.items():
                col = np.asarray(bucket[name]).copy()
                col[surplus] = value
                bucket[name] = jnp.asarray(col)
            conns[lvl] = bucket
        state = dataclasses.replace(state, conns=tuple(conns))
    # Growth reads the batch-mean state under batching: reduce the recorded
    # per-sample activations with px's own batch reduction.
    if "batch_activations" in doc:
        from plastax.phases import batch_mean_units

        per_sample = jnp.asarray(doc["batch_activations"], jnp.float32)
        batched = {
            name: jnp.broadcast_to(col, (per_sample.shape[0], *col.shape))
            for name, col in state.units.items()
        }
        batched[px.ACTIVATION.name] = per_sample
        state = dataclasses.replace(state, units=batch_mean_units(batched))
    else:
        act = jnp.asarray([u["fields"]["activation"] for u in units], dtype=jnp.float32)
        state = dataclasses.replace(
            state, units={**state.units, px.ACTIVATION.name: act}
        )
    step = int(trigger.get("step", params["step"]))
    state = dataclasses.replace(
        state,
        step=jnp.int32(step),
        units_added=jnp.int32(trigger.get("units_added_this_step", 0)),
    )
    return _Net, static, state


def _grow_once(
    doc: dict[str, Any], *, step_cap: int | None, free_at: dict[int, int]
) -> tuple[list[tuple[int, int, float]], Any]:
    from plastax.phases import build_add_conn_phase

    net, static, state = _score_net(doc, step_cap=step_cap, free_at=free_at)
    before = _live_pairs(state)
    phase = build_add_conn_phase(net, static)
    new_state, _ = phase(state, px.StepInputs(inputs=jnp.zeros((0,)), targets=None))
    grown = sorted(_live_pairs(new_state))
    for pair in before:
        grown.remove(pair)
    assert int(new_state.grown) == len(grown), "state.grown miscounts the commits"
    return grown, new_state


@pytest.mark.parametrize("name", sorted(_ENFORCED_SCORE))
def test_grow_score_golden(name: str) -> None:
    """The score pipeline reproduces the reference commit set and flags exactly."""
    doc = _load(f"{name}.json")
    free_slots = int(doc["params"]["free_slots"])
    expect = doc["expect"]
    want = sorted(
        (
            int(c["src"]),
            int(c["dst"]),
            float(np.float32((((3 * c["src"] + 5 * c["dst"]) % 16) - 8) / 8.0)),
        )
        for c in expect["committed"]
    )
    level = {u["id"]: u["level"] for u in doc["initial_units"]}

    # Run A: px's whole selection, nothing capacity-bound.
    selected, state_a = _grow_once(doc, step_cap=None, free_at={})
    assert not bool(state_a.overflow), "ample capacity must not overflow"
    assert (len(selected) > free_slots) == expect["conn_overflow"], (
        f"{name}: px selected {len(selected)} for {free_slots} free slots"
    )

    # Run B: the reference's single claim domain as a step cap.
    grown, state_b = _grow_once(doc, step_cap=free_slots, free_at={})
    assert grown == want, f"{name}: committed edges diverge from the reference"
    assert len(grown) == expect["grown"]
    assert bool(state_b.needs_resort) == expect["needs_resort"]

    # Run C: px's real per-bucket claim, slots spent as the prefix spends them.
    per_level: dict[int, int] = {}
    for s_id, _, _ in selected:
        per_level[level[s_id]] = per_level.get(level[s_id], 0) + 1
    free_at, left = {}, free_slots
    for lvl in sorted(per_level):
        free_at[lvl] = min(left, per_level[lvl])
        left -= free_at[lvl]
    grown_c, state_c = _grow_once(doc, step_cap=None, free_at=free_at)
    assert grown_c == want, f"{name}: the real claim diverges"
    assert bool(state_c.overflow) == expect["conn_overflow"]
    assert bool(state_c.needs_resort) == expect["needs_resort"]


# ---------------------------------------------------------------------------
# unit_lifecycle_v1, update cases: the real update_unit phase, exactly.
# ---------------------------------------------------------------------------


class _UnitUpdateV1(px.UpdateUnit[None]):
    """unit_update_v1: activation += 1/8."""

    def update(self, u: px.UnitView, i: px.UnitIdx, g: None) -> px.UnitWrite:
        """Add 1/8 to the activation."""
        del g
        return px.UnitWrite.of((px.ACTIVATION, u[px.ACTIVATION, i] + 0.125))


@pytest.mark.parametrize("name", sorted(_ENFORCED_UNIT))
def test_unit_update_golden(name: str) -> None:
    """The update_unit phase reproduces the reference unit state exactly."""
    import dataclasses

    from plastax.phases import build_update_unit_phase

    doc = _load(f"{name}.json")
    assert doc["rules"] == {"update_unit": "unit_update_v1"}
    assert doc["field_defaults"] == {px.ACTIVATION.name: 0.0}
    units = doc["initial_units"]
    n = len(units)

    class _Net(px.Network[None]):
        forward_pass = ReluForward()
        update_unit = _UnitUpdateV1()
        unit_capacity = int(doc["capacity"])
        max_levels = int(doc["max_levels"])
        propagation = px.Propagation.TOPOLOGICAL

    edges = doc["initial_edges"]
    static, state = px.NetworkBuilder.from_edges(
        _Net,
        n,
        np.asarray([e["src"] for e in edges], dtype=np.int32),
        np.asarray([e["dst"] for e in edges], dtype=np.int32),
        weights=np.asarray([e["fields"]["weight"] for e in edges], dtype=np.float32),
        input_ids=[u["id"] for u in units if u["is_input"]],
        output_ids=[u["id"] for u in units if u["is_output"]],
        globals_=None,
    )
    assert static.num_units == doc["capacity"]
    # The golden's allocated slots carry their own field values and pruned
    # flags (the builder cannot mark an allocated unit pruned).
    ids = jnp.asarray([u["id"] for u in units], jnp.int32)
    cols = dict(state.units)
    cols[px.ACTIVATION.name] = (
        cols[px.ACTIVATION.name]
        .at[ids]
        .set(jnp.asarray([u["fields"]["activation"] for u in units], jnp.float32))
    )
    cols[px.PRUNED.name] = (
        cols[px.PRUNED.name].at[ids].set(jnp.asarray([u["pruned"] for u in units]))
    )
    cols[px.LEVEL.name] = (
        cols[px.LEVEL.name]
        .at[ids]
        .set(jnp.asarray([u["level"] for u in units], jnp.int32))
    )
    state = dataclasses.replace(state, units=cols)
    before = np.asarray(state.units[px.ACTIVATION.name]).copy()

    phase = build_update_unit_phase(_Net, static)
    new_state, _ = phase(state, px.StepInputs(inputs=jnp.zeros((0,)), targets=None))

    act = np.asarray(new_state.units[px.ACTIVATION.name])
    pruned = np.asarray(new_state.units[px.PRUNED.name])
    for want in doc["expect"]["units"]:
        uid = want["id"]
        assert act[uid] == want["fields"]["activation"], f"unit {uid} activation"
        assert bool(pruned[uid]) == want["pruned"], f"unit {uid} pruned"
    # The never-allocated slots are free and untouched.
    assert pruned[n:].all()
    np.testing.assert_array_equal(act[n:], before[n:])
    assert doc["expect"]["pruned"] == []
    want_edges = sorted(
        (int(e["src"]), int(e["dst"]), float(e["fields"]["weight"]))
        for e in doc["expect"]["edges"]
    )
    assert _live_pairs(new_state) == want_edges
