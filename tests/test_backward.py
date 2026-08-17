"""Backward sweep (M3).

Direction reversal: accumulate into the source unit
(dispatch_cpu.hpp:232-258). Implemented when M3 lands.
"""

import pytest

pytestmark = pytest.mark.skip(reason="pending M3")


def test_backward_placeholder() -> None:
    raise NotImplementedError
