"""Pytree contract for NetworkState / NetworkStatic (M1).

NetworkState flatten/unflatten roundtrip; NetworkStatic meta fields hash/eq;
changing a meta field changes the PyTreeDef; changing a leaf does not.
Implemented when M1 lands (see tests/README.md, IMPLEMENTATION_PLAN.md).
"""

import pytest

pytestmark = pytest.mark.skip(reason="pending M1")


def test_pytree_placeholder() -> None:
    raise NotImplementedError
