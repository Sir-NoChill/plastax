"""Versioned rule definitions used by the cross-library goldens.

Each rule here has a name of the form ``<rule>_v<N>`` recorded in every golden
that uses it. A library consuming such a golden implements the same rule under
the same name (plastax-cpp: ``tests/parity/golden_rules.hpp``) and must never
change its semantics in place -- a semantic change is a new version, new
goldens, and a deliberate commit.

All scores, weights and field values are integers or dyadic fractions, so
float32 arithmetic on them is exact and goldens compare with ``==``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import reference as ref

# --- weights ---------------------------------------------------------------


def dyadic_weight_v1(src: int, dst: int) -> float:
    """w(src, dst) = (((3*src + 5*dst) mod 16) - 8) / 8."""
    return float(np.float32((((3 * src + 5 * dst) % 16) - 8) / 8.0))


# --- passes ----------------------------------------------------------------
# relu_mlp_v1: forward map = w * act_src, combine = sum, apply = relu;
# loss = MSE (grad staged to loss_grad); backward map = w * grad_pre_act_dst,
# combine = sum, apply: grad_pre_act = (acc + loss_grad) * [act > 0].

RELU_MLP_V1 = "relu_mlp_v1"


# --- growth scores ---------------------------------------------------------


def grid_score_v1(src: int, dst: int) -> float:
    """score(src, dst) = (((3*src + 5*dst) mod 17) - 8) / 8.

    mod 17 over small grids produces deliberate score ties, exercising the
    total order's (src, dst, index) tie-breaks.
    """
    return float(np.float32((((3 * src + 5 * dst) % 17) - 8) / 8.0))


def importance_v1(uid: int) -> float:
    """importance(i) = ((7*i) mod 13) / 4, with deliberate ties."""
    return float(np.float32(((7 * uid) % 13) / 4.0))


def predicate_v1(src: int, dst: int) -> bool:
    """should_add(src, dst) iff src + dst is even."""
    return (src + dst) % 2 == 0


def predicate_score_v1(src: int, dst: int) -> float:
    """The predicate adapter's score: 0 for should_add, else -inf."""
    return 0.0 if predicate_v1(src, dst) else ref.NEG_INF


# --- growth proposals (consume the site rng; draw order is part of the spec)


def hash_propose_v1(
    n_units: int,
) -> Any:
    """Per-unit proposer: dst = uniform_int(n) [sub 0], score [sub 1].

    score = floor(uniform * 256) / 256, a dyadic fraction in [0, 1).
    src is the proposing unit.
    """

    def propose(u: dict[str, Any], j: int, rng: ref.SiteRng) -> tuple[int, int, float]:
        del j
        dst = rng.uniform_int(n_units)
        score = float(np.float32(int(np.float32(rng.uniform()) * 256) / 256.0))
        return u["id"], dst, score

    return propose


def conn_propose_v1(n_units: int) -> Any:
    """Per-connection proposer.

    src = conn.src, dst = uniform_int(n) [sub 0], score = floor(uniform * 256)
    / 256 [sub 1].
    """

    def propose(
        edge: dict[str, Any],
        key: tuple[int, int, int],
        j: int,
        rng: ref.SiteRng,
    ) -> tuple[int, int, float]:
        del key, j
        dst = rng.uniform_int(n_units)
        score = float(np.float32(int(np.float32(rng.uniform()) * 256) / 256.0))
        return edge["src"], dst, score

    return propose


def global_propose_v1(n_units: int) -> Any:
    """Global proposer: src [sub 0], dst [sub 1], score [sub 2]."""

    def propose(j: int, rng: ref.SiteRng) -> tuple[int, int, float]:
        del j
        src = rng.uniform_int(n_units)
        dst = rng.uniform_int(n_units)
        score = float(np.float32(int(np.float32(rng.uniform()) * 256) / 256.0))
        return src, dst, score

    return propose


def grow_init_v1(src: int, dst: int) -> dict[str, float]:
    """New-edge init: weight = dyadic_weight_v1(src, dst)."""
    return {"weight": dyadic_weight_v1(src, dst)}


def score_act_v1(units: list[dict[str, Any]]) -> Any:
    """score(src, dst) = activation[src] + activation[dst] (state-dependent).

    Used by the batched golden: under batching, growth reads the batch-mean
    state, so the score changes with the mean.
    """

    def score(src: int, dst: int) -> float:
        a = np.float32(units[src]["fields"]["activation"])
        b = np.float32(units[dst]["fields"]["activation"])
        return float(np.float32(a + b))

    return score


# --- unit rules ------------------------------------------------------------


def unit_update_v1(u: dict[str, Any]) -> dict[str, float]:
    """Activation += 1/8 (exact in float32 for these magnitudes)."""
    return {
        "activation": float(
            np.float32(np.float32(u["fields"]["activation"]) + np.float32(0.125))
        )
    }


def unit_prune_v1(u: dict[str, Any]) -> bool:
    """Prune iff activation < -1/2."""
    return u["fields"]["activation"] < -0.5


def unit_spawn_v1(u: dict[str, Any]) -> tuple[bool, int]:
    """Spawn one child (level offset +1) iff activation >= 1/2."""
    return u["fields"]["activation"] >= 0.5, 1


def unit_spawn_deep_v1(u: dict[str, Any]) -> tuple[bool, int]:
    """Spawn with a huge offset (+30000) iff activation >= 1/2 (clamp case)."""
    return u["fields"]["activation"] >= 0.5, 30000


def unit_child_init_v1(
    child: dict[str, Any], parent: dict[str, Any]
) -> dict[str, float]:
    """Child activation = parent activation / 2."""
    del child
    return {
        "activation": float(np.float32(parent["fields"]["activation"]) / np.float32(2))
    }
