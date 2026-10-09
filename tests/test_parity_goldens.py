"""Consume the registry goldens (`tests/golden/*.json`, schema registry_v1).

The goldens are emitted by `scripts/parity/emit.py` from the pure-NumPy
reference in `scripts/parity/reference.py`; plastax-cpp consumes the same
files. Each golden names the feature set it `requires`:

- ``passes_v1`` runs here today, with exact float equality (every value in
  those goldens is a dyadic fraction, so float32 arithmetic on them is exact).
- ``unit_lifecycle_v1`` and ``growth_v2`` are the specification of phases this
  library does not implement yet; their goldens are skipped loudly below, one
  visible skip per file, until the features land.

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
_IMPLEMENTED = {"passes_v1"}
_SPEC_ONLY = {
    "unit_lifecycle_v1": "the unit lifecycle (update/prune/add) is not implemented",
    "growth_v2": "the shared growth selection pipeline is not implemented",
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
    """L = 0.5*(pred - target)^2 per output; stages dL/dpred to loss_grad."""

    def per_output(
        self, u: px.UnitView, i: px.UnitIdx, target: jax.Array, g: None
    ) -> tuple[jax.Array, px.UnitWrite]:
        """Return the loss contribution and stage the gradient."""
        del g
        diff = u[px.ACTIVATION, i] - target
        return jnp.float32(0.5) * diff * diff, px.UnitWrite.of((LOSS_GRAD, diff))


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
# Spec-only goldens: loud per-file skips until the features land.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _registry_goldens(), ids=lambda p: p.stem)
def test_registry_golden_is_consumed_or_knowingly_skipped(path: pathlib.Path) -> None:
    """Every registry golden is either consumed above or skipped by name."""
    doc = json.loads(path.read_text())
    requires = doc["requires"]
    if requires in _IMPLEMENTED:
        assert doc["name"] in {
            "passes_relu_topological",
            "passes_relu_pipeline",
        }, f"{doc['name']} claims {requires} but no consumer covers it"
        return
    assert requires in _SPEC_ONLY, f"unknown requires tag {requires!r} in {path.name}"
    pytest.skip(f"{doc['name']}: {_SPEC_ONLY[requires]} ({requires})")
