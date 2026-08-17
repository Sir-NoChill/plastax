"""Donation contract (M5).

Donation warning promoted to error (pytest filterwarnings); the step is
shape-preserving on every leaf of the state pytree. Implemented when M5 lands.
"""

import pytest

pytestmark = pytest.mark.skip(reason="pending M5")


def test_donation_placeholder() -> None:
    raise NotImplementedError
