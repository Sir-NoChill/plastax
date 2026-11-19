"""Drawing a record: every unit and every connection, for looking at small networks.

matplotlib is an optional dependency (`plastax[plot]`) and is imported only when
`draw` is called. Nothing here imports JAX or any other part of plastax.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from plastax.layers._record import Record

if TYPE_CHECKING:
    from matplotlib.axes import Axes

_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_CREATED = "#e4e3df"  # fill of a unit a layer created; inputs and constants are plain
# Line colours by block name. A name not listed here takes the next of `_MORE` in the
# order the names first appear, and `_REST` once those run out.
_BY_NAME = {"weight": "#2a78d6", "bias": "#eb6834"}
_MORE = ("#1baf7a", "#eda100", "#e87ba4")
_REST = "#8a8984"
# Above this many units the ids are not written inside the circles.
_LABEL_LIMIT = 200


def _depths(record: Record) -> np.ndarray:
    """Give each unit the length of the longest path of connections leading to it.

    Args:
        record: The record.

    Returns:
        `(num_units,)` depths. A unit nothing connects to has depth 0.

    Raises:
        ValueError: If the connections close a cycle.
    """
    depth = np.zeros(record.num_units, np.int64)
    if not record.blocks:
        return depth
    src = np.concatenate([block.src for block in record.blocks])
    dst = np.concatenate([block.dst for block in record.blocks])
    # Each round pushes every depth one connection further. A longest path has
    # fewer than `num_units` connections, so more rounds than that means a cycle.
    for _ in range(record.num_units):
        deeper = depth.copy()
        np.maximum.at(deeper, dst, depth[src] + 1)
        if np.array_equal(deeper, depth):
            return depth
        depth = deeper
    raise ValueError("a record whose connections close a cycle cannot be drawn")


def _layout(record: Record) -> np.ndarray:
    """Place every unit of a record on a plane.

    A unit's column is its depth. Within a column, units run from top to bottom in
    the order of their ids, centred on the horizontal axis. A constant unit is not
    in a column: it sits half a column before the units it feeds, below both
    neighbouring columns.

    Args:
        record: The record.

    Returns:
        `(num_units, 2)` coordinates, x then y, one column or one row apart.
    """
    depth = _depths(record)
    constant = np.zeros(record.num_units, bool)
    constant[record.constants] = True
    xy = np.zeros((record.num_units, 2))

    height = np.zeros(int(depth.max(initial=0)) + 2, np.int64)
    for d in range(height.size - 1):
        units = np.flatnonzero((depth == d) & ~constant)
        height[d] = units.size
        xy[units, 0] = d
        xy[units, 1] = (units.size - 1) / 2 - np.arange(units.size)

    # Constants that feed the same column are set side by side below it.
    below: dict[int, int] = {}
    for unit in record.constants.tolist():
        fed = np.concatenate(
            [block.dst[block.src == unit] for block in record.blocks]
            or [np.zeros(0, np.int32)]
        )
        column = int(depth[fed].min()) if fed.size else 1
        k = below.get(column, 0)
        below[column] = k + 1
        tallest = max(height[column - 1], height[column])
        xy[unit] = (column - 0.5, -(tallest - 1) / 2 - 1.5 - k)
    return xy


def draw(record: Record, *, ax: Axes | None = None, max_units: int = 500) -> Axes:
    """Draw every unit and every connection of a record.

    Units are circles holding their record id: inputs plain, units a layer created
    shaded, constant units square. Connections are lines, coloured by block name.
    Columns run from the inputs on the left by depth, the longest path of
    connections leading to a unit.

    For looking at small networks while building them. Needs matplotlib, which
    `plastax[plot]` installs.

    Args:
        record: The record to draw.
        ax: The axes to draw into. A new figure is made when this is None.
        max_units: The most units the record may have. A drawing with a line per
            connection stops being readable well below the size of a real network.

    Returns:
        The axes drawn into.

    Raises:
        ImportError: If matplotlib is not installed.
        ValueError: If the record has more than `max_units` units, or its
            connections close a cycle.
    """
    if record.num_units > max_units:
        raise ValueError(
            f"this record has {record.num_units} units and `draw` is limited to "
            f"{max_units}; pass a larger `max_units` to draw it anyway"
        )
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.lines import Line2D
    except ImportError as error:
        raise ImportError(
            "plastax.layers.draw needs matplotlib; install it with `plastax[plot]`"
        ) from error

    xy = _layout(record)
    if ax is None:
        columns = float(np.ptp(xy[:, 0])) if record.num_units else 0.0
        rows = float(np.ptp(xy[:, 1])) if record.num_units else 0.0
        size = (max(4.0, 2.4 * columns + 2.5), max(3.0, 0.42 * rows + 1.8))
        _, ax = plt.subplots(figsize=size, facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)

    colours: dict[str, str] = {}
    counts: dict[str, int] = {}
    for block in record.blocks:
        if block.name not in colours:
            spare = len([name for name in colours if name not in _BY_NAME])
            colours[block.name] = _BY_NAME.get(
                block.name, _MORE[spare] if spare < len(_MORE) else _REST
            )
        counts[block.name] = counts.get(block.name, 0) + block.src.size
        ax.add_collection(
            LineCollection(
                list(np.stack([xy[block.src], xy[block.dst]], axis=1)),
                colors=colours[block.name],
                linewidths=1.1,
                alpha=0.75,
                capstyle="round",
                zorder=1,
            )
        )

    constant = np.zeros(record.num_units, bool)
    constant[record.constants] = True
    given = np.zeros(record.num_units, bool)
    for ids in record.inputs.values():
        given[ids.reshape(-1)] = True
    labelled = record.num_units <= _LABEL_LIMIT
    area = 330 if labelled else 60
    for mask, marker, face in (
        (given, "o", _SURFACE),
        (~given & ~constant, "o", _CREATED),
        (constant, "s", _SURFACE),
    ):
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=area,
            marker=marker,
            facecolor=face,
            edgecolor=_INK,
            linewidth=1.1,
            zorder=2,
        )
    if labelled:
        for unit, (x, y) in enumerate(xy):
            ax.text(
                x, y, str(unit), ha="center", va="center", fontsize=7.5, color=_INK
            ).set_zorder(3)

    if colours:
        ax.legend(
            handles=[
                Line2D([], [], color=colour, lw=2, label=f"{name} ({counts[name]})")
                for name, colour in colours.items()
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=len(colours),
            frameon=False,
            labelcolor=_INK,
        )
    if record.num_units:
        ax.set_xlim(xy[:, 0].min() - 0.6, xy[:, 0].max() + 0.6)
        ax.set_ylim(xy[:, 1].min() - 0.9, xy[:, 1].max() + 0.9)
    ax.axis("off")
    return ax
