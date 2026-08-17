"""Builder -> finalize invariants (M1).

Levels correct; buckets sorted by (dead, to_id); capacities obey
capacity_policy. Implemented when M1 lands.
"""

import pytest

pytestmark = pytest.mark.skip(reason="pending M1")


def test_builder_placeholder() -> None:
    raise NotImplementedError
