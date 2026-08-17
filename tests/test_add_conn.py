"""AddConn (M4).

K-bounded candidates, top_k selection, prefix-sum slot claim, overflow flag,
and level-preserving adds do not set needs_resort. Implemented when M4 lands.
"""

import pytest

pytestmark = pytest.mark.skip(reason="pending M4")


def test_add_conn_placeholder() -> None:
    raise NotImplementedError
