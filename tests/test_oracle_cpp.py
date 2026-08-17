"""C++ oracle parity (M5).

Golden-file parity vs the C++ examples (tolerance-based; see plan section
"Oracle harness"). Implemented when M5 lands.
"""

import pytest

pytestmark = pytest.mark.skip(reason="pending M5")


def test_oracle_cpp_placeholder() -> None:
    raise NotImplementedError
