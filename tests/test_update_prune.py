"""UpdateConn / PruneConn (M4).

UpdateConn incoming/outgoing two-pass ordering; PruneConn tombstoning; derived
live counts. Implemented when M4 lands.
"""

import pytest

pytestmark = pytest.mark.skip(reason="pending M4")


def test_update_prune_placeholder() -> None:
    raise NotImplementedError
