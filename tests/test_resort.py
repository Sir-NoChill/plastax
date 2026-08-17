"""Resort + retrace contract (M4).

recompute_levels vs host Kahn; resort produces sorted compacted buckets;
retrace count == 1 per resort via
jax.test_util.assert_num_jit_and_pmap_compilations; a pure add/prune workload
compiles exactly once. Implemented when M4 lands.
"""

import pytest

pytestmark = pytest.mark.skip(reason="pending M4")


def test_resort_placeholder() -> None:
    raise NotImplementedError
