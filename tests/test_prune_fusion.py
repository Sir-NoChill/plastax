"""Prune-into-forward fusion: the legality check and fused == two-pass.

`make_step(fuse_prune=...)` may evaluate the prune predicate inside the
forward sweep (`phases.plan_prune_fusion`). On CPU the fused step is the XLA
lowering (`fuse_prune="xla"`); the single-pass Triton kernel needs an NVIDIA
GPU and is checked by `examples/benchmarks/fused_prune_check.py`. These tests
pin that the fused step's whole state (tombstones, free-slot claims, every
column) is bit-identical to the two-pass step's over many churn steps, and
that a predicate reading something the forward (or a phase between forward
and prune) writes is never fused.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import plastax as px
from plastax import phases

_WIDTH = 24
_LAYERS = 3
_NUM_UNITS = _WIDTH * _LAYERS
_EDGES_PER_LAYER = 200
_STEPS = 40

G = dict[str, jax.Array]

# Written by the forward's apply from the globals alone (no accumulator), and
# read by the bench-style prune predicate: legal, "forwarded".
PRUNED = px.FieldSpec.int32("test/pruned")
# Written by the forward's apply from the accumulator: never legal to read.
FIRED = px.FieldSpec.boolean("test/fired", default=False)
TRACE = px.FieldSpec.float32("test/trace")


def _hash01(a: jax.Array, b: jax.Array, c: jax.Array) -> jax.Array:
    h = (a.astype(jnp.uint32) + jnp.uint32(0x9E3779B1)) * jnp.uint32(0x85EBCA77)
    h = (h ^ b.astype(jnp.uint32)) * jnp.uint32(0xC2B2AE3D)
    h = (h ^ c.astype(jnp.uint32)) * jnp.uint32(0x27D4EB2F)
    h = h ^ (h >> 15)
    return (h >> jnp.uint32(8)).astype(jnp.float32) / jnp.float32(1 << 24)


def _marked(i: jax.Array, g: G) -> jax.Array:
    """Whether unit i is on this step's prune list (a sorted, -1-padded row)."""
    ids = g["prune"][g["step"] % g["prune"].shape[0]]
    pos = jnp.minimum(
        jnp.searchsorted(ids, i, method="scan_unrolled"), ids.shape[0] - 1
    )
    return ids[pos] == i


class _MarkForward(px.ForwardPass):
    combine = px.monoid.sum_
    linear_input = px.ACTIVATION

    def map(self, u: Any, dst: Any, src: Any, c: Any, cid: Any, g: G) -> jax.Array:
        del dst, g
        return c[px.WEIGHT, cid] * u[px.ACTIVATION, src]

    def apply(self, u: Any, i: Any, g: G, acc: jax.Array) -> px.UnitWrite:
        del u
        return px.UnitWrite.of(
            (px.ACTIVATION, jnp.tanh(acc)),
            (PRUNED, _marked(i, g).astype(jnp.int32)),
            (FIRED, acc > 0.5),
        )


class _MarkedPrune(px.PruneConn):
    """Kill every edge touching a marked unit (the synth-bench predicate)."""

    def predicate(self, u: Any, c: Any, cid: Any, g: G) -> jax.Array:
        del g
        return (u[PRUNED, c[px.FROM_ID, cid]] | u[PRUNED, c[px.TO_ID, cid]]) == 1


class _HashPrune(px.PruneConn):
    """Kill a hashed fraction of edges each step: reads only edge columns + g."""

    def predicate(self, u: Any, c: Any, cid: Any, g: G) -> jax.Array:
        del u
        return _hash01(c[px.FROM_ID, cid], c[px.TO_ID, cid], g["step"]) < 0.15


class _FiredPrune(px.PruneConn):
    """Reads a field the forward computes from its accumulator: not fusable."""

    def predicate(self, u: Any, c: Any, cid: Any, g: G) -> jax.Array:
        del g
        return u[FIRED, c[px.TO_ID, cid]]


class _ActivationPrune(px.PruneConn):
    """Reads ACTIVATION, the forward's main output: not fusable."""

    def predicate(self, u: Any, c: Any, cid: Any, g: G) -> jax.Array:
        del g
        return u[px.ACTIVATION, c[px.FROM_ID, cid]] < -0.9


class _WeightPrune(px.PruneConn):
    """Magnitude prune: reads WEIGHT."""

    def predicate(self, u: Any, c: Any, cid: Any, g: G) -> jax.Array:
        del u, g
        return jnp.abs(c[px.WEIGHT, cid]) < 0.05


class _TracePrune(px.PruneConn):
    """Reads TRACE, which the loss writes."""

    def predicate(self, u: Any, c: Any, cid: Any, g: G) -> jax.Array:
        del g
        return u[TRACE, c[px.TO_ID, cid]] > 10.0


class _Grow(px.AddConn):
    max_candidates = 48

    def score(self, u: Any, src: Any, dst: Any, g: G) -> jax.Array:
        deeper = u[px.LEVEL, dst] == u[px.LEVEL, src] + 1
        return jnp.where(deeper, _hash01(src, dst, g["step"] + 977), -jnp.inf)

    def init(self, u: Any, src: Any, dst: Any, g: G) -> px.ConnWrite:
        del u
        return px.ConnWrite.of((px.WEIGHT, _hash01(dst, src, g["step"]) - 0.5))


class _Propose(px.ProposeAddConn):
    """Re-grow random deeper edges (the synth-bench growth, hashed)."""

    max_candidates = 64
    num_proposals = 2 * _NUM_UNITS

    def propose(self, u: Any, j: jax.Array, g: G) -> tuple[Any, Any, Any]:
        src = j // 2
        dst = (_hash01(j, g["step"], jnp.int32(5)) * _NUM_UNITS).astype(jnp.int32)
        deeper = u[px.LEVEL, dst] == u[px.LEVEL, src] + 1
        return src, dst, jnp.where(deeper, _hash01(dst, j, g["step"]), -jnp.inf)

    def init(self, u: Any, src: Any, dst: Any, g: G) -> px.ConnWrite:
        del u
        return px.ConnWrite.of((px.WEIGHT, _hash01(src, dst, g["step"]) - 0.5))


class _Tick(px.ResetGlobal):
    def reset(self, g: G) -> G:
        return {**g, "step": g["step"] + 1}


class _MseLoss(px.Loss):
    def per_output(self, u: Any, i: Any, target: Any, g: G) -> tuple[Any, Any]:
        del g
        err = u[px.ACTIVATION, i] - target
        return 0.5 * err * err, px.UnitWrite.of((TRACE, err))


class _Backward(px.BackwardPass):
    combine = px.monoid.sum_

    def map(self, u: Any, src: Any, dst: Any, c: Any, cid: Any, g: G) -> jax.Array:
        del src, g
        return c[px.WEIGHT, cid] * u[TRACE, dst]

    def apply(self, u: Any, i: Any, g: G, acc: jax.Array) -> px.UnitWrite:
        del u, i, g
        return px.UnitWrite.of((TRACE, acc))


class _Sgd(px.UpdateConn):
    def incoming(self, u: Any, dst: Any, src: Any, c: Any, cid: Any, g: G) -> Any:
        del g
        w = c[px.WEIGHT, cid] - 0.01 * u[TRACE, dst] * u[px.ACTIVATION, src]
        return px.ConnWrite.of((px.WEIGHT, w))

    def outgoing(self, u: Any, src: Any, dst: Any, c: Any, cid: Any, g: G) -> Any:
        del u, src, dst, g
        return px.ConnWrite.of((px.WEIGHT, c[px.WEIGHT, cid]))


def _net(
    prune: px.PruneConn,
    *,
    add: Any = None,
    train: bool = False,
    propagation: px.Propagation = px.Propagation.TOPOLOGICAL,
) -> type[px.Network[G]]:
    mode = propagation

    class Net(px.Network[G]):
        forward_pass = _MarkForward()
        loss = _MseLoss() if train else None
        backward_pass = _Backward() if train else None
        update_conn = _Sgd() if train else None
        prune_conn = prune
        add_conn = add
        reset_global = _Tick()
        extra_unit_fields = (PRUNED, FIRED, TRACE)
        propagation = mode

    return Net


def _globals(rng: np.random.Generator) -> G:
    rows = np.where(
        rng.random((16, 5)) < 0.8, rng.integers(_WIDTH, _NUM_UNITS, (16, 5)), -1
    )
    return {
        "step": jnp.int32(0),
        "prune": jnp.asarray(np.sort(rows, axis=1).astype(np.int32)),
    }


def _build(
    net: type[px.Network[G]], *, align: int | None = None
) -> tuple[px.NetworkStatic, px.NetworkState[G]]:
    rng = np.random.default_rng(3)
    src, dst = [], []
    for layer in range(_LAYERS - 1):
        ids = rng.choice(_WIDTH * _WIDTH, _EDGES_PER_LAYER, replace=False)
        src.append(layer * _WIDTH + ids // _WIDTH)
        dst.append((layer + 1) * _WIDTH + ids % _WIDTH)
    frm = np.concatenate(src).astype(np.int32)
    to = np.concatenate(dst).astype(np.int32)
    return px.NetworkBuilder.from_edges(
        net,
        _NUM_UNITS,
        frm,
        to,
        weights=rng.standard_normal(frm.shape[0]).astype(np.float32),
        input_ids=list(range(_WIDTH)),
        output_ids=list(range((_LAYERS - 1) * _WIDTH, _NUM_UNITS)),
        globals_=_globals(rng),
        capacity_headroom=0.5,
        capacity_align=align,
    )


def _copy(state: px.NetworkState[G]) -> px.NetworkState[G]:
    return jax.tree.map(jnp.copy, state)


def _run_both(
    net: type[px.Network[G]], *, steps: int = _STEPS, align: int | None = None
) -> tuple[px.NetworkState[G], px.NetworkState[G], Any]:
    static, state = _build(net, align=align)
    fused = px.make_step(net, static, fuse_prune="xla")
    plain = px.make_step(net, static, fuse_prune="off")
    rng = np.random.default_rng(7)
    a, b = _copy(state), _copy(state)
    for _ in range(steps):
        x = jnp.asarray(rng.standard_normal(_WIDTH).astype(np.float32))
        t = (
            jnp.asarray(rng.standard_normal(_WIDTH).astype(np.float32))
            if net.loss is not None
            else None
        )
        inputs = px.StepInputs(inputs=x, targets=t)
        ra, rb = fused(a, inputs), plain(b, inputs)
        assert bool(ra.overflow) == bool(rb.overflow)
        _assert_same(ra.loss, rb.loss, "loss")
        a, b = ra.state, rb.state
    return a, b, getattr(fused, "prune_fusion").plan  # noqa: B009


def _assert_same(la: Any, lb: Any, msg: str = "") -> None:
    # Bit-identical on CPU. A GPU scatter-add sums in no fixed order, so there
    # float columns match to rounding; every integer and bool column (ids,
    # tombstones, counts) must still be exact.
    la, lb = np.asarray(la), np.asarray(lb)
    if la.dtype.kind == "f" and jax.default_backend() != "cpu":
        np.testing.assert_allclose(la, lb, rtol=1e-5, atol=1e-5, err_msg=msg)
    else:
        np.testing.assert_array_equal(la, lb, err_msg=msg)


def _assert_identical(a: px.NetworkState[G], b: px.NetworkState[G]) -> None:
    leaves_a = jax.tree_util.tree_flatten_with_path(a)[0]
    leaves_b = jax.tree.leaves(b)
    for (path, la), lb in zip(leaves_a, leaves_b, strict=True):
        _assert_same(la, lb, jax.tree_util.keystr(path))


@pytest.mark.parametrize(
    ("name", "net"),
    [
        ("marked+propose", _net(_MarkedPrune(), add=_Propose())),
        ("marked+grid", _net(_MarkedPrune(), add=_Grow())),
        ("hash+grid", _net(_HashPrune(), add=_Grow())),
        ("hash+train", _net(_HashPrune(), add=_Grow(), train=True)),
        ("marked, no growth", _net(_MarkedPrune())),
        (
            "marked+propose, pipeline",
            _net(_MarkedPrune(), add=_Propose(), propagation=px.Propagation.PIPELINE),
        ),
    ],
)
def test_fused_step_matches_two_pass_over_churn(
    name: str, net: type[px.Network[G]]
) -> None:
    a, b, plan = _run_both(net)
    assert plan.fused and plan.engine == "xla", (name, plan)
    _assert_identical(a, b)
    assert int(px.state.live_conn_count(a)) > 0, name


def test_fused_step_reuses_free_counts_on_the_two_level_claim() -> None:
    # Capacities that are multiples of the free-slot block, with a small claim
    # (k * block <= capacity), so add_conn takes the two-level search.
    class _Small(_Grow):
        max_candidates = 1

    net = _net(_MarkedPrune(), add=_Small())
    a, b, plan = _run_both(net, align=1024)
    assert plan.fused
    _assert_identical(a, b)


def test_marked_predicate_forwards_the_flag() -> None:
    net = _net(_MarkedPrune(), add=_Propose())
    static, state = _build(net)
    plan = phases.plan_prune_fusion(net, static, state.globals_, engine="xla")
    assert plan.fused and plan.forwarded == (PRUNED.name,)


@pytest.mark.parametrize(
    ("prune", "train", "why"),
    [
        (_ActivationPrune(), False, "accumulator"),
        (_FiredPrune(), False, "accumulator"),
        (_WeightPrune(), True, "weight"),
        (_TracePrune(), True, TRACE.name),
    ],
)
def test_predicate_reading_a_written_field_is_not_fused(
    prune: px.PruneConn, train: bool, why: str
) -> None:
    net = _net(prune, add=_Grow(), train=train)
    static, state = _build(net)
    for engine in ("auto", "triton", "xla"):
        plan = phases.plan_prune_fusion(net, static, state.globals_, engine=engine)
        assert not plan.fused, plan
        assert why in plan.reason, plan
    step = px.make_step(net, static, fuse_prune="xla")
    x = jnp.zeros((_WIDTH,), jnp.float32)
    t = jnp.zeros((_WIDTH,), jnp.float32) if train else None
    step(_copy(state), px.StepInputs(inputs=x, targets=t))
    record = getattr(step, "prune_fusion")  # noqa: B009
    assert record.plan is not None and not record.plan.fused


def test_weight_prune_without_update_conn_is_fused() -> None:
    # Reading WEIGHT is fine when nothing between forward and prune writes it.
    net = _net(_WeightPrune(), add=_Grow())
    a, b, plan = _run_both(net, steps=10)
    assert plan.fused
    _assert_identical(a, b)


class _SelfReadForward(_MarkForward):
    """Writes PRUNED from FIRED -- a column the forward itself writes."""

    def apply(self, u: Any, i: Any, g: G, acc: jax.Array) -> px.UnitWrite:
        del g
        return px.UnitWrite.of(
            (px.ACTIVATION, acc),
            (FIRED, acc > 0.0),
            (PRUNED, u[FIRED, i].astype(jnp.int32)),
        )


def test_forwarded_field_must_not_read_a_forward_written_column() -> None:
    class Net(px.Network[G]):
        forward_pass = _SelfReadForward()
        prune_conn = _MarkedPrune()
        reset_global = _Tick()
        extra_unit_fields = (PRUNED, FIRED, TRACE)

    static, state = _build(Net)
    plan = phases.plan_prune_fusion(Net, static, state.globals_, engine="xla")
    assert not plan.fused and FIRED.name in plan.reason, plan


def test_auto_keeps_two_passes_without_the_triton_kernel() -> None:
    net = _net(_MarkedPrune(), add=_Propose())
    static, state = _build(net)
    plan = phases.plan_prune_fusion(net, static, state.globals_)
    if not phases.nvidia_triton_available():
        assert not plan.fused and "two-pass" in plan.reason
    off = phases.plan_prune_fusion(_net(_MarkedPrune()), static, state.globals_)
    assert off.engine in (None, "triton")


def test_off_and_unknown_engine() -> None:
    net = _net(_MarkedPrune(), add=_Propose())
    static, state = _build(net)
    step = px.make_step(net, static, fuse_prune="off")
    step(_copy(state), px.StepInputs(inputs=jnp.zeros((_WIDTH,)), targets=None))
    record = getattr(step, "prune_fusion")  # noqa: B009
    assert record.plan is not None and not record.plan.fused
    with pytest.raises((ValueError, TypeError), match="fuse_prune"):
        px.make_step(net, static, fuse_prune="sometimes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="engine"):
        phases.plan_prune_fusion(net, static, state.globals_, engine="sometimes")


def test_no_prune_conn_is_not_fused() -> None:
    class Net(px.Network[G]):
        forward_pass = _MarkForward()
        extra_unit_fields = (PRUNED, FIRED, TRACE)

    static, state = _build(Net)
    plan = phases.plan_prune_fusion(Net, static, state.globals_, engine="xla")
    assert not plan.fused and plan.reason == "no prune_conn"


def _translate(prune: px.PruneConn, net: type[px.Network[G]]) -> Any:
    static, state = _build(net)
    units = {
        s.name: jax.ShapeDtypeStruct((static.num_units,), s.dtype)
        for s in static.unit_fields
    }
    cap = static.level_capacities[0]
    conns = {s.name: jax.ShapeDtypeStruct((cap,), s.dtype) for s in static.conn_fields}
    args = (units, conns, jax.ShapeDtypeStruct((), jnp.int32), state.globals_)

    def pred(u: Any, c: Any, cid: Any, g: Any) -> Any:
        return prune.predicate(px.UnitView(u), px.ConnView(c), cid, g)

    closed = jax.make_jaxpr(pred)(*args)
    return phases._PredicateTranslator(closed, phases._flat_labels(args)).build()


def test_translator_accepts_exact_integer_predicates() -> None:
    marked = _translate(_MarkedPrune(), _net(_MarkedPrune()))
    assert ("unit", PRUNED.name) in marked.inputs
    # The edge's own from/to come from the kernel's registers, not new loads.
    assert all(kind != "conn" for kind, _ in marked.inputs)
    hashed = _translate(_HashPrune(), _net(_HashPrune()))
    assert "tl.load" in hashed.source and ">>" in hashed.source


class _ScaledPrune(px.PruneConn):
    def predicate(self, u: Any, c: Any, cid: Any, g: G) -> jax.Array:
        del u, g
        return c[px.WEIGHT, cid] * 3.0 < 0.1  # a rounding float multiply


def test_translator_rejects_inexact_float_arithmetic() -> None:
    with pytest.raises(phases._Untranslatable, match="mul"):
        _translate(_ScaledPrune(), _net(_ScaledPrune()))


@pytest.mark.skipif(
    not phases.nvidia_triton_available(), reason="needs an NVIDIA GPU + jax_triton"
)
@pytest.mark.parametrize("align", [256, 1024])
@pytest.mark.parametrize(
    ("name", "net"),
    [
        ("marked+propose", _net(_MarkedPrune(), add=_Propose())),
        ("hash+grid", _net(_HashPrune(), add=_Grow())),
        ("hash+train", _net(_HashPrune(), add=_Grow(), train=True)),
        (
            "marked+propose, pipeline",
            _net(_MarkedPrune(), add=_Propose(), propagation=px.Propagation.PIPELINE),
        ),
    ],
)
def test_fused_counts_feed_the_triton_claim(
    name: str, net: type[px.Network[G]], align: int, monkeypatch: Any
) -> None:
    # The fused kernel's 256-slot free counts go straight into triton_claim:
    # no count reduction is traced, and the state matches the two-pass step
    # with the XLA claim (integer and bool columns exactly; floats to the
    # forward's summation order).
    static, state = _build(net, align=align)
    recounts: list[int] = []
    counting = phases.free_block_counts

    def spy(dead: jax.Array, max_block: int = 1024) -> jax.Array:
        recounts.append(max_block)
        return counting(dead, max_block)

    monkeypatch.setattr(phases, "free_block_counts", spy)
    wired = px.make_step(net, static, fuse_prune="triton", growth="triton")
    plain = px.make_step(net, static, fuse_prune="off", growth="xla")
    rng = np.random.default_rng(7)
    a, b = _copy(state), _copy(state)
    for i in range(_STEPS):
        x = jnp.asarray(rng.standard_normal(_WIDTH).astype(np.float32))
        t = (
            jnp.asarray(rng.standard_normal(_WIDTH).astype(np.float32))
            if net.loss is not None
            else None
        )
        inputs = px.StepInputs(inputs=x, targets=t)
        ra = wired(a, inputs)
        if i == 0:
            assert recounts == [], (name, recounts)  # traced: nothing recounted
            plan = getattr(wired, "prune_fusion").plan  # noqa: B009
            assert plan.fused and plan.engine == "triton", (name, plan)
        rb = plain(b, inputs)
        assert bool(ra.overflow) == bool(rb.overflow), (name, i)
        a, b = ra.state, rb.state
    leaves_a = jax.tree_util.tree_flatten_with_path(a)[0]
    for (path, la), lb in zip(leaves_a, jax.tree.leaves(b), strict=True):
        la, lb = np.asarray(la), np.asarray(lb)
        msg = f"{name}: {jax.tree_util.keystr(path)}"
        if la.dtype.kind == "f":
            np.testing.assert_allclose(la, lb, rtol=1e-5, atol=1e-5, err_msg=msg)
        else:
            np.testing.assert_array_equal(la, lb, err_msg=msg)
    assert int(px.state.live_conn_count(a)) > 0, name
