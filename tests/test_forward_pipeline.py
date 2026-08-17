"""Pipeline forward sweep (M2).

Flat sweep vs numpy reference; one-hop latency semantics
(dispatch_cpu.hpp:202-223); dead-slot null-scatter. Implemented when M2 lands.
"""

import pytest

pytestmark = pytest.mark.skip(reason="pending M2")


def test_forward_pipeline_placeholder() -> None:
    raise NotImplementedError
