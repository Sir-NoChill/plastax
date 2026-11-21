"""Policies over the network's globals."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ResetGlobalPolicy[GS](Protocol):
    """Global-state reset policy invoked between episodes or runs.

    Type Args:
        GS: the global state type threaded through the network.
    """

    def reset(self, g: GS) -> GS:
        """Reset the global state.

        Args:
            g: the current global state.

        Returns:
            The reset global state.
        """
        ...
