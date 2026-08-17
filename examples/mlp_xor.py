"""Port of examples/mlp-xor/mlp_xor.cpp: sigmoid MLP, MSE loss, SGD.

Canonical backprop trait example and the first oracle target: topological
mode, all four differentiable phases, no dynamics.
"""
from __future__ import annotations

import numpy as np

import plastax as px

GradPreAct = px.FieldSpec.f32("grad_pre_act")


class SigmoidForward(px.ForwardPass):
    combine = px.monoid.sum_
    # map: activation[src] * weight; apply: sigmoid(acc), stash pre-act grad
    def map(self, u, dst, src, c, cid, g):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def apply(self, u, i, g, acc):  # type: ignore[no-untyped-def]
        raise NotImplementedError


class XorNet(px.Network[None]):
    forward_pass = SigmoidForward()
    # backward_pass, loss (MSE), update_conn (SGD) per mlp_xor.cpp
    extra_unit_fields = (GradPreAct,)
    propagation = px.Propagation.TOPOLOGICAL


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
