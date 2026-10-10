"""ScoreAddConn's rule surface: knobs the registry goldens do not pin.

The growth_v2 goldens (test_parity_goldens.py) enforce the score pipeline
against the reference. These tests cover what they cannot: knobs whose
golden does not bind (the threshold golden selects exactly what top_k
would), class-definition validation, the removed attribute names,
`on_overflow = "error"`, `Network.structural_interval`, the predicate
adapter's overrides, and `candidates = "shortlist_per_level"` (no golden
yet), checked against the reference selection over L14-built candidates.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import plastax as px
from plastax import phases

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts/parity"))
import reference as ref  # noqa: E402

_INPUTS = px.StepInputs(inputs=jnp.zeros((0,), dtype=jnp.float32), targets=None)

# Three levels of three units: 0-2 inputs, 3-5 hidden, 6-8 outputs.
_LEVELS = (0, 0, 0, 1, 1, 1, 2, 2, 2)
_EDGES = ((0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8))


class _SumForward(px.ForwardPass):
    combine = px.monoid.sum_

    def map(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: Any,
    ) -> jax.Array:
        return c[px.WEIGHT, cid] * u[px.ACTIVATION, src]

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: Any, acc: jax.Array
    ) -> px.UnitWrite:
        return px.UnitWrite.of((px.ACTIVATION, acc))


def _hash_score(src: jax.Array, dst: jax.Array) -> jax.Array:
    """(((3*src + 5*dst) mod 17) - 8) / 8: dyadic, with deliberate ties."""
    return (((3 * src + 5 * dst) % 17) - 8).astype(jnp.float32) / jnp.float32(8.0)


def _importance(i: jax.Array) -> jax.Array:
    """(5*i) mod 4: heavy ties, so the ascending-id tiebreak decides."""
    return ((5 * i) % 4).astype(jnp.float32)


class _Base:
    """Hash-scored rule; tests subclass it to set knobs."""

    max_new_per_level = 2

    def score(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: Any
    ) -> jax.Array:
        del u, g
        return _hash_score(src, dst)

    def importance(self, u: px.UnitView, i: px.UnitIdx, g: Any) -> jax.Array:
        del u, g
        return _importance(i)

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: Any
    ) -> px.ConnWrite:
        del u, src, dst, g
        return px.ConnWrite.of((px.WEIGHT, jnp.float32(1.0)))


def _net(rule: object, **attrs: Any) -> type[px.Network[Any]]:
    body: dict[str, Any] = {
        "forward_pass": _SumForward(),
        "add_conn": rule,
        "propagation": px.Propagation.TOPOLOGICAL,
        "batch_reduction": px.MeanFloatFirstRest(),
        **attrs,
    }
    return type("_Net", (px.Network,), body)


def _build(net: type[px.Network[Any]], globals_: Any = None) -> tuple[Any, Any]:
    static, state = px.NetworkBuilder.from_edges(
        net,
        len(_LEVELS),
        np.asarray([s for s, _ in _EDGES], dtype=np.int32),
        np.asarray([d for _, d in _EDGES], dtype=np.int32),
        weights=np.ones(len(_EDGES), np.float32),
        input_ids=[0, 1, 2],
        output_ids=[6, 7, 8],
        globals_=globals_,
        capacity_headroom=4.0,
    )
    assert tuple(np.asarray(state.units[px.LEVEL.name]).tolist()) == _LEVELS
    return static, state


def _live(state: Any) -> list[tuple[int, int]]:
    out = []
    for bucket in state.conns:
        dead = np.asarray(bucket[px.DEAD.name])
        src = np.asarray(bucket[px.FROM_ID.name])
        dst = np.asarray(bucket[px.TO_ID.name])
        out += [
            (int(s), int(d)) for s, d, x in zip(src, dst, dead, strict=True) if not x
        ]
    return sorted(out)


def _grown(net: type[px.Network[Any]], globals_: Any = None) -> list[tuple[int, int]]:
    static, state = _build(net, globals_)
    new_state, _ = phases.build_add_conn_phase(net, static)(state, _INPUTS)
    grown = _live(new_state)
    for pair in _live(state):
        grown.remove(pair)
    assert int(new_state.grown) == len(grown)
    return sorted(grown)


# --- selection = "threshold": read from g, per step -------------------------


class _Threshold(_Base):
    selection = "threshold"
    max_new_per_level = 9
    max_level_gap = 2

    def threshold(self, g: Any) -> jax.Array:
        return g["t"]


def test_threshold_binds_and_is_read_from_globals() -> None:
    net = _net(_Threshold())
    low = _grown(net, {"t": jnp.float32(-1.0)})
    high = _grown(net, {"t": jnp.float32(0.5)})
    # A binding threshold keeps strictly fewer, every one scoring >= 0.5.
    assert 0 < len(high) < len(low)
    assert all(float(_hash_score(jnp.int32(s), jnp.int32(d))) >= 0.5 for s, d in high)
    # ...and drops nothing that clears it (capped at 9 per level, not reached).
    assert set(high) == {
        (s, d) for s, d in low if float(_hash_score(jnp.int32(s), jnp.int32(d))) >= 0.5
    }


# --- class-definition validation ---------------------------------------------


@pytest.mark.parametrize(
    ("removed", "names"),
    [
        ("dedupe", ("dedupe_live", "dedupe_step")),
        ("max_candidates", ("max_new_per_level",)),
        ("max_candidate_units", ("shortlist_size", "candidates")),
        ("shortlist_per_level", ("shortlist_per_level",)),
    ],
)
def test_removed_attribute_names_its_replacement(
    removed: str, names: tuple[str, ...]
) -> None:
    rule = _Base()
    setattr(rule, removed, True)
    with pytest.raises(TypeError) as err:
        _net(rule)
    for name in names:
        assert name in str(err.value)


@pytest.mark.parametrize(
    ("attrs", "match"),
    [
        ({"selection": "best"}, "selection must be one of"),
        ({"max_new_per_level": None}, "max_new_per_level must be an int"),
        ({"selection": "threshold"}, "requires a `threshold"),
        ({"direction": "up"}, "direction must be one of"),
        ({"trigger": ("every", 0)}, "trigger must be"),
        ({"trigger": "when"}, "requires a `when"),
        ({"on_overflow": "warn"}, "on_overflow must be"),
        ({"max_new_per_step": 0}, "max_new_per_step must be"),
        ({"allow_self_loops": 1}, "allow_self_loops must be a bool"),
        ({"candidates": "grid"}, "candidates must be one of"),
        ({"candidates": "shortlist"}, "shortlist_size must be an int"),
    ],
)
def test_bad_knobs_are_rejected_at_class_definition(
    attrs: dict[str, Any], match: str
) -> None:
    rule = type("_Rule", (_Base,), attrs)()
    with pytest.raises(TypeError, match=match):
        _net(rule)


def test_selection_all_needs_no_max_new_per_level() -> None:
    rule = type("_Rule", (_Base,), {"selection": "all", "max_new_per_level": None})()
    assert len(_grown(_net(rule))) > 0


def test_shortlist_requires_importance() -> None:
    class _NoImportance:
        max_new_per_level = 1
        candidates = "shortlist"
        shortlist_size = 2
        score = _Base.score
        init = _Base.init

    with pytest.raises(TypeError, match="importance"):
        _net(_NoImportance())


@pytest.mark.parametrize("interval", [0, True, 1.5])
def test_bad_structural_interval_is_rejected(interval: object) -> None:
    with pytest.raises(TypeError, match="structural_interval"):
        _net(_Base(), structural_interval=interval)


# --- on_overflow = "error" ----------------------------------------------------


def test_on_overflow_error_raises_and_flag_does_not() -> None:
    every = {"selection": "all", "max_new_per_level": None, "max_level_gap": 2}
    flagging = _net(type("_Flag", (_Base,), every)())
    static, state = _build(flagging)
    # Leave bucket 0 a single free slot so its selection overflows.
    bucket = dict(state.conns[0])
    dead = np.asarray(bucket[px.DEAD.name]).copy()
    free = np.flatnonzero(dead)
    dead[free[1:]] = False
    bucket[px.DEAD.name] = jnp.asarray(dead)
    import dataclasses

    tight = dataclasses.replace(state, conns=(bucket, *state.conns[1:]))
    flagged, _ = phases.build_add_conn_phase(flagging, static)(tight, _INPUTS)
    assert bool(flagged.overflow)

    erroring = _net(type("_Err", (_Base,), {**every, "on_overflow": "error"})())
    with pytest.raises(Exception, match="overflow"):
        out, _ = phases.build_add_conn_phase(erroring, static)(tight, _INPUTS)
        jax.block_until_ready(out.conns)


# --- Network.structural_interval ---------------------------------------------


def test_structural_interval_gates_growth_to_every_nth_step() -> None:
    net = _net(
        type("_All", (_Base,), {"selection": "all", "max_new_per_level": None})(),
        structural_interval=3,
    )
    static, state = _build(net)
    step = px.make_step(net, static)
    fired = []
    for _ in range(7):
        state = step(state, px.StepInputs(inputs=jnp.zeros((3,)), targets=None)).state
        fired.append(int(state.grown) > 0)
    # Steps 0..6 run with the pre-increment counter: growth on 0, 3 and 6.
    # (Step 3 and 6 still find candidates: the rule never dedupes.)
    assert fired == [True, False, False, True, False, False, True]


class _PruneAll:
    """Tombstone every live connection."""

    def predicate(
        self, u: px.UnitView, c: px.ConnView, cid: px.ConnIdx, g: Any
    ) -> jax.Array:
        del u, c, cid, g
        return jnp.bool_(True)


@pytest.mark.parametrize(
    ("propagation", "batch_size", "fuse_prune"),
    [
        (px.Propagation.TOPOLOGICAL, None, "off"),
        (px.Propagation.TOPOLOGICAL, None, "xla"),
        (px.Propagation.TOPOLOGICAL, 2, "off"),
        (px.Propagation.PIPELINE, None, "off"),
    ],
)
def test_structural_interval_does_not_gate_connection_pruning(
    propagation: px.Propagation, batch_size: int | None, fuse_prune: Any
) -> None:
    """With interval 3, prune_conn runs every step; growth only on 0 and 3.

    prune_conn runs before add_conn within a step, so the edges a growth step
    adds survive that step and the next step's pruning removes them.
    """
    net = _net(
        type("_All", (_Base,), {"selection": "all", "max_new_per_level": None})(),
        structural_interval=3,
        prune_conn=_PruneAll(),
        propagation=propagation,
    )
    static, state = _build(net)
    step = px.make_step(net, static, batch_size=batch_size, fuse_prune=fuse_prune)
    shape = (3,) if batch_size is None else (batch_size, 3)
    grew, alive = [], []
    for _ in range(6):
        result = step(state, px.StepInputs(inputs=jnp.zeros(shape), targets=None))
        state = result.state
        grew.append(int(state.grown) > 0)
        alive.append(len(_live(state)) > 0)
    if fuse_prune == "xla":
        assert step.prune_fusion.plan.fused
    assert grew == [True, False, False, True, False, False]
    assert alive == grew


# --- predicate_add_conn -------------------------------------------------------


def test_predicate_adapter_scores_and_overrides() -> None:
    def even(u: px.UnitView, s: px.UnitIdx, d: px.UnitIdx, g: Any) -> jax.Array:
        del u, g
        return (s + d) % 2 == 0

    rule = px.predicate_add_conn(
        even, _Base().init, max_level_gap=2, direction="deeper"
    )
    assert isinstance(rule, px.ScoreAddConn)
    assert rule.selection == "all" and rule.dedupe_step is True  # type: ignore[attr-defined]
    grown = _grown(_net(rule))
    expected = {
        (s, d)
        for s in range(9)
        for d in range(9)
        if (s + d) % 2 == 0 and _LEVELS[d] > _LEVELS[s]
    }
    assert set(grown) == expected


# --- candidates = "shortlist_per_level" (L14) ---------------------------------


def _l14_candidates(m: int, gap: int, direction: str) -> list[dict[str, Any]]:
    """L14: per source level (ascending), the level's top-M sources by
    importance x the top-M window-eligible destinations, index
    level_rank*M*M + row-major; importance ties by ascending id."""

    def eligible(ls: int, ld: int) -> bool:
        if abs(ld - ls) > gap:
            return False
        if direction == "deeper":
            return ld > ls
        if direction == "same_or_deeper":
            return ld >= ls
        return True

    def ranked(pred: Any) -> list[int]:
        ids = [i for i in range(len(_LEVELS)) if pred(i)]
        return sorted(ids, key=lambda i: (-float(_importance(jnp.int32(i))), i))[:m]

    out = []
    for lr, lvl in enumerate(sorted(set(_LEVELS))):
        srcs = ranked(lambda i, lvl=lvl: _LEVELS[i] == lvl)
        dsts = ranked(lambda i, lvl=lvl: eligible(lvl, _LEVELS[i]))
        for i, s in enumerate(srcs):
            for k, d in enumerate(dsts):
                out.append(
                    {
                        "src": s,
                        "dst": d,
                        "score": float(_hash_score(jnp.int32(s), jnp.int32(d))),
                        "index": lr * m * m + i * len(dsts) + k,
                    }
                )
    return out


@pytest.mark.parametrize("direction", ["any", "deeper", "same_or_deeper"])
def test_shortlist_per_level_matches_the_reference_selection(direction: str) -> None:
    m, gap = 2, 1
    rule = type(
        "_PerLevel",
        (_Base,),
        {
            "candidates": "shortlist_per_level",
            "shortlist_size": m,
            "max_level_gap": gap,
            "direction": direction,
        },
    )()
    units = [{"id": i, "level": lvl, "pruned": False} for i, lvl in enumerate(_LEVELS)]
    edges = [{"src": s, "dst": d} for s, d in _EDGES]
    want = ref.select_growth(
        units,
        edges,
        _l14_candidates(m, gap, direction),
        capacity=16,
        max_level_gap=gap,
        direction=direction,
        max_new_per_level=_Base.max_new_per_level,
    )
    got = _grown(_net(rule))
    assert want, "the case must select something to be meaningful"
    assert got == sorted((c["src"], c["dst"]) for c in want)


def test_shortlist_per_level_is_topological_only() -> None:
    rule = type(
        "_PerLevel",
        (_Base,),
        {"candidates": "shortlist_per_level", "shortlist_size": 2},
    )()
    net = _net(rule, propagation=px.Propagation.PIPELINE)
    static, _ = _build(_net(_Base()))
    with pytest.raises(ValueError, match="topological-only"):
        phases.build_add_conn_phase(net, static)
