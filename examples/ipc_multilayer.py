"""Port of examples/ipc-multilayer/ipc_multilayer.cpp: streaming iPC,
pipeline mode. The flagship pipeline example; oracle target for M4.

C++ traits being mirrored:
  ForwardPass=iPCForwardPass, BackwardPass=iPCBackwardPass,
  UpdateConn=iPCUpdateConn,
  ExtraUnitFields={ValueNode, Error, BottomUp}, Model=Pipeline.
"""
from __future__ import annotations

import plastax as px

ValueNode = px.FieldSpec.f32("value_node")
Error = px.FieldSpec.f32("error")
BottomUp = px.FieldSpec.f32("bottom_up")


class IpcNet(px.Network[None]):
    # forward_pass, backward_pass, update_conn per ipc_multilayer.cpp
    extra_unit_fields = (ValueNode, Error, BottomUp)
    propagation = px.Propagation.PIPELINE


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
