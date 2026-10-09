"""Emit the plastax-cpp conformance vectors.

Runs each network in `parity_vectors.py` and writes what it produced, step by
step, to `tests/golden/*.json` in the plastax-cpp checkout (located through
`PLASTAX_CPP_DIR`; see `tests/_plastax_cpp.py`). plastax-cpp's `test_parity_goldens.cpp`
rebuilds the same networks in C++ and checks it agrees within tolerance.

plastax is the oracle here: these files are the specification, not a record of
two implementations happening to agree. Regenerating one redefines what plastax-cpp
must do, so it is a deliberate act -- never automatic, never part of CI. A
golden diff in a pull request should be read as an intentional change to the
contract.

Usage::

    uv run python scripts/emit_parity_goldens.py            # all vectors
    uv run python scripts/emit_parity_goldens.py --only mlp_optim_adam
    uv run python scripts/emit_parity_goldens.py --check    # fail if stale

`--check` regenerates in memory and diffs against what is on disk without
writing, which is what CI should run if it ever wants to catch a golden that
drifted from its generator.

See `notes/parity/00-parity-harness.md` in plastax-cpp.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import subprocess
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import plastax as px

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "tests"))
from parity_vectors import VECTORS, Vector, build_topology  # noqa: E402

from _plastax_cpp import plastax_cpp_dir  # noqa: E402

_REPO = pathlib.Path(__file__).resolve().parents[1]
_GOLDEN_DIR = plastax_cpp_dir() / "tests" / "golden"


def _plastax_commit() -> str:
    """Return the current plastax commit, or a marker when it cannot be read.

    The commit is recorded in every golden so a file that drifted from its
    generator is visible in review rather than inferred.

    Returns:
        The short commit sha, with "-dirty" appended when the tree is modified.
    """
    try:
        sha = subprocess.run(
            ["git", "-C", str(_REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(_REPO), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _live_edges(static: px.NetworkStatic, state: Any) -> list[dict[str, Any]]:
    """Collect every live edge as {from, to, w}, sorted by (from, to).

    Weights are keyed by unit pair rather than by index on purpose: plastax
    buckets connections by level and sorts within a bucket, so its connection
    ids do not correspond to plastax-cpp's allocation order. The pair is the only
    identifier both implementations agree on.

    Args:
        static: The network's static configuration.
        state: The current network state.

    Returns:
        One dict per live edge, sorted for a stable diff.
    """
    edges: list[dict[str, Any]] = []
    for bucket in range(len(static.level_capacities)):
        conns = state.conns[bucket]
        dead = np.asarray(conns[px.DEAD.name])
        src = np.asarray(conns[px.FROM_ID.name])
        dst = np.asarray(conns[px.TO_ID.name])
        weight = np.asarray(conns[px.WEIGHT.name])
        for i in range(dead.shape[0]):
            if bool(dead[i]):
                continue
            edges.append(
                {"from": int(src[i]), "to": int(dst[i]), "w": float(weight[i])}
            )
    edges.sort(key=lambda e: (e["from"], e["to"]))
    return edges


def _run(vector: Vector) -> dict[str, Any]:
    """Build and drive one vector, returning its golden document.

    Args:
        vector: The vector to run.

    Returns:
        The complete golden document, ready to serialise.

    Raises:
        RuntimeError: If a step reports arena overflow, which none of these
            static networks should ever do.
    """
    static, state = px.NetworkBuilder.from_topology(
        vector.net, build_topology(vector), jax.random.PRNGKey(0), globals_=None
    )

    # No vector mutates structure, so the Driver's overflow/resort protocol is
    # not needed; calling the jitted step directly also surfaces the loss,
    # which Driver.step does not return.
    step = px.make_step(vector.net, static)

    levels = [int(v) for v in np.asarray(state.units[px.LEVEL.name])]
    network: dict[str, Any] = {
        "traits": vector.traits,
        "input_dim": vector.input_dim,
        "layers": [
            {"units": layer.units, "init": layer.init.as_json()}
            for layer in vector.layers
        ],
        "propagation": vector.net.propagation.name.lower(),
    }
    hyper: dict[str, float] = dict(vector.hyper)
    if vector.learning_rate is not None:
        hyper["learning_rate"] = vector.learning_rate
    if hyper:
        network["hyper"] = hyper

    doc: dict[str, Any] = {
        "name": vector.name,
        "description": vector.description,
        "generated_by": {
            "repo": "plastax",
            "commit": _plastax_commit(),
            "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "generator": "scripts/emit_parity_goldens.py",
        },
        "network": network,
        "expect_structure": {
            "num_units": int(static.num_units),
            "num_conns": int(px.state.live_conn_count(state)),
            "levels": levels,
        },
        # Compared exactly, not within tolerance: identical initial weights are
        # the precondition that makes every later comparison interpretable, and
        # the RNG port is bit-exact (tests/test_plastax_cpp_rng.py).
        "expect_initial_weights": _live_edges(static, state),
        "tolerance": {
            "default": {"rtol": vector.rtol, "atol": vector.atol},
        },
        "steps": [],
    }

    for index, (inputs, targets) in enumerate(vector.steps):
        step_inputs = px.StepInputs(
            inputs=jnp.asarray(inputs, dtype=jnp.float32),
            targets=None
            if targets is None
            else jnp.asarray(targets, dtype=jnp.float32),
        )
        result = step(state, step_inputs)
        if bool(result.overflow):
            raise RuntimeError(
                f"{vector.name}: step {index} overflowed its arena; a static "
                "conformance vector must never grow"
            )
        state = result.state

        doc["steps"].append(
            {
                "inputs": [float(v) for v in inputs],
                "targets": None if targets is None else [float(v) for v in targets],
                "expect": {
                    "activations": [
                        float(v) for v in np.asarray(state.units[px.ACTIVATION.name])
                    ],
                    "weights": _live_edges(static, state),
                    "live_edges": int(px.state.live_conn_count(state)),
                    # Emitted for reference but not yet asserted: plastax-cpp's Loss
                    # returns void and stages only the gradient, so there is no
                    # loss value to compare. notes/parity/05-loss-split.md adds
                    # Network::GetLastLoss(); the C++ runner picks it up then.
                    "loss": float(result.loss),
                },
            }
        )
    return doc


def _serialise(doc: dict[str, Any]) -> str:
    """Render a golden document as stable, reviewable JSON."""
    return json.dumps(doc, indent=2, sort_keys=False) + "\n"


def main() -> int:
    """Generate (or check) the golden vectors.

    Returns:
        0 on success, 1 when --check finds a stale or missing golden.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="regenerate a single vector by name")
    parser.add_argument(
        "--check",
        action="store_true",
        help="report stale goldens without writing them",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=_GOLDEN_DIR,
        help=f"output directory (default: {_GOLDEN_DIR})",
    )
    args = parser.parse_args()

    vectors = [v for v in VECTORS if args.only in (None, v.name)]
    if not vectors:
        known = ", ".join(v.name for v in VECTORS)
        print(f"no vector named {args.only!r}; known: {known}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    stale: list[str] = []
    for vector in vectors:
        text = _serialise(_run(vector))
        path = args.out / f"{vector.name}.json"

        if args.check:
            # Ignore the provenance block when diffing: its timestamp changes
            # on every run and would report every golden as stale.
            current = json.loads(path.read_text()) if path.is_file() else None
            fresh = json.loads(text)
            if current is not None:
                current.pop("generated_by", None)
                fresh.pop("generated_by", None)
            if current != fresh:
                stale.append(vector.name)
                print(f"STALE  {vector.name}")
            else:
                print(f"ok     {vector.name}")
            continue

        path.write_text(text)
        steps = len(vector.steps)
        print(f"wrote  {path.relative_to(args.out.parent.parent)}  ({steps} steps)")

    if stale:
        print(
            f"\n{len(stale)} golden(s) differ from the generator. Regenerate with "
            "`uv run python scripts/emit_parity_goldens.py` and review the diff.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
