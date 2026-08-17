"""Phase elision (M2).

Absent phases produce identical jaxprs to a hand-assembled subset (compare
jax.make_jaxpr output structure). Implemented when M2 lands.
"""

import pytest

pytestmark = pytest.mark.skip(reason="pending M2")


def test_phases_elision_placeholder() -> None:
    raise NotImplementedError
