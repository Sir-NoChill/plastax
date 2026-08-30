"""Extreme Classification Repository data: reader, cache, and precision@k.

The repository's BoW/TF-IDF files (Bhatia et al.,
http://manikvarma.org/downloads/XC/XMLRepository.html) are plain text::

    num_points num_features num_labels
    label,label,... feat:val feat:val ...
    ...

with the label list empty (a leading space) for a point carrying no labels.
Both sides are sparse, so a point is stored CSR-style -- `indptr` bounds into
flat `indices`/`values` arrays -- rather than as dense rows: Amazon-670K is
490k points over 135k features, which densifies to ~265 GB but is ~5 GB sparse.

Parsing the text is the slow part (single-pass Python over ~1 GB), so
`load_xmc` caches the parsed arrays to a sibling ``.npz`` and reuses it.
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np


@dataclasses.dataclass(frozen=True)
class XmcSplit:
    """One CSR-style split of an extreme-classification dataset.

    Attributes:
        num_features: dimensionality of the BoW/TF-IDF feature space.
        num_labels: size of the label set.
        x_indptr: (num_points+1,) row bounds into x_indices/x_values.
        x_indices: flat feature ids, grouped by point.
        x_values: flat feature values, parallel to x_indices.
        y_indptr: (num_points+1,) row bounds into y_indices.
        y_indices: flat label ids, grouped by point.
    """

    num_features: int
    num_labels: int
    x_indptr: np.ndarray
    x_indices: np.ndarray
    x_values: np.ndarray
    y_indptr: np.ndarray
    y_indices: np.ndarray

    @property
    def num_points(self) -> int:
        """Number of points in the split."""
        return len(self.x_indptr) - 1

    def features(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the (feature ids, values) of point ``i``."""
        lo, hi = int(self.x_indptr[i]), int(self.x_indptr[i + 1])
        return self.x_indices[lo:hi], self.x_values[lo:hi]

    def labels(self, i: int) -> np.ndarray:
        """Return the label ids of point ``i``."""
        lo, hi = int(self.y_indptr[i]), int(self.y_indptr[i + 1])
        return self.y_indices[lo:hi]

    def dense_input(self, i: int, *, bias: float = 1.0) -> np.ndarray:
        """Scatter point ``i`` into a dense (num_features+1,) input vector.

        The trailing element is the constant bias input (mlp_xor.py's third
        input unit): it lets every downstream unit learn a bias through an
        ordinary incoming weight, with no extra plumbing.

        Args:
            i: point index.
            bias: value of the trailing constant input unit.

        Returns:
            A dense float32 vector over the input units.
        """
        out = np.zeros((self.num_features + 1,), dtype=np.float32)
        indices, values = self.features(i)
        out[indices] = values
        out[self.num_features] = bias
        return out


def parse_xmc(path: str | pathlib.Path) -> XmcSplit:
    """Parse one Extreme Classification Repository text file.

    Args:
        path: path to the ``*_train.txt`` / ``*_test.txt`` file.

    Returns:
        The parsed split.

    Raises:
        ValueError: if the header line is not ``num_points num_features
            num_labels``.
    """
    x_indices: list[np.ndarray] = []
    x_values: list[np.ndarray] = []
    y_indices: list[np.ndarray] = []
    x_counts: list[int] = []
    y_counts: list[int] = []
    with open(path, encoding="utf-8") as handle:
        header = handle.readline().split()
        if len(header) != 3:
            raise ValueError(f"{path}: bad header {header!r}")
        _, num_features, num_labels = (int(v) for v in header)
        for line in handle:
            # split(" ", 1) not .split(): the label field is EMPTY for a point
            # with no labels, and bare .split() would silently promote that
            # line's first feature pair into the label slot.
            label_field, _, feature_field = line.partition(" ")
            labels = (
                np.fromstring(label_field, dtype=np.int32, sep=",")
                if label_field
                else np.zeros((0,), dtype=np.int32)
            )
            pairs = feature_field.split()
            row_i = np.empty((len(pairs),), dtype=np.int32)
            row_v = np.empty((len(pairs),), dtype=np.float32)
            for k, pair in enumerate(pairs):
                key, _, value = pair.partition(":")
                row_i[k] = int(key)
                row_v[k] = float(value)
            x_indices.append(row_i)
            x_values.append(row_v)
            y_indices.append(labels)
            x_counts.append(len(row_i))
            y_counts.append(len(labels))

    def _indptr(counts: list[int]) -> np.ndarray:
        out = np.zeros((len(counts) + 1,), dtype=np.int64)
        np.cumsum(counts, out=out[1:])
        return out

    empty_i, empty_f = np.zeros((0,), np.int32), np.zeros((0,), np.float32)
    return XmcSplit(
        num_features=num_features,
        num_labels=num_labels,
        x_indptr=_indptr(x_counts),
        x_indices=np.concatenate(x_indices) if x_indices else empty_i,
        x_values=np.concatenate(x_values) if x_values else empty_f,
        y_indptr=_indptr(y_counts),
        y_indices=np.concatenate(y_indices) if y_indices else empty_i,
    )


def load_xmc(path: str | pathlib.Path, *, use_cache: bool = True) -> XmcSplit:
    """Parse a repository split, caching the arrays to a sibling ``.npz``.

    Args:
        path: path to the ``*.txt`` split file.
        use_cache: read (and write) the ``.npz`` cache beside ``path``.

    Returns:
        The parsed split.
    """
    path = pathlib.Path(path)
    cache = path.with_suffix(".npz")
    if use_cache and cache.exists():
        with np.load(cache) as data:
            return XmcSplit(
                num_features=int(data["num_features"]),
                num_labels=int(data["num_labels"]),
                x_indptr=data["x_indptr"],
                x_indices=data["x_indices"],
                x_values=data["x_values"],
                y_indptr=data["y_indptr"],
                y_indices=data["y_indices"],
            )
    split = parse_xmc(path)
    if use_cache:
        np.savez(
            cache,
            num_features=split.num_features,
            num_labels=split.num_labels,
            x_indptr=split.x_indptr,
            x_indices=split.x_indices,
            x_values=split.x_values,
            y_indptr=split.y_indptr,
            y_indices=split.y_indices,
        )
    return split


def precision_at_k(
    scores: np.ndarray, labels: np.ndarray, ks: tuple[int, ...]
) -> tuple[float, ...]:
    """Precision@k of one point's label scores against its true label set.

    P@k is the fraction of the k highest-scoring labels that are truly
    relevant -- the standard extreme-classification metric, because with
    10^5-10^6 labels and a handful positive per point, accuracy is vacuous.

    Args:
        scores: (num_labels,) predicted score per label.
        labels: the point's true label ids.
        ks: the cutoffs to evaluate.

    Returns:
        One precision value per entry of ``ks``.
    """
    if len(labels) == 0:
        return tuple(0.0 for _ in ks)
    top = np.argpartition(-scores, min(max(ks), len(scores) - 1))[: max(ks)]
    top = top[np.argsort(-scores[top])]
    hits = np.isin(top, labels)
    return tuple(float(np.sum(hits[:k])) / k for k in ks)
