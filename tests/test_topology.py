"""Topology generators (M1).

dense edge count = n_in*n_out; conv2d edge enumeration matches
lax.conv_general_dilated shape semantics (positions, receptive fields,
stride); initializer statistics sane; sequential id offsetting; from_topology
equals the equivalent manual builder calls. Implemented when M1 lands.
"""

import pytest

pytestmark = pytest.mark.skip(reason="pending M1")


def test_topology_placeholder() -> None:
    raise NotImplementedError
