"""Locate the plastax-cpp checkout (the C++ implementation).

Cross-library artifacts -- the RNG golden, the conformance vectors -- live in
the plastax-cpp tree. Its location comes from the ``PLASTAX_CPP_DIR``
environment variable, defaulting to a ``plastax-cpp`` or ``plastix`` (the
upstream repository's name) directory beside this repository. Tests that need
the checkout skip when it is absent; scripts fail loudly instead, since
silently emitting goldens nowhere helps nobody.
"""

from __future__ import annotations

import os
import pathlib

_REPO = pathlib.Path(__file__).resolve().parents[1]


def plastax_cpp_dir() -> pathlib.Path:
    """Path to the plastax-cpp checkout.

    Returns:
        ``$PLASTAX_CPP_DIR`` if set, else the first of ``plastax-cpp`` and
        ``plastix`` beside this repository that exists (``plastax-cpp`` when
        neither does). The path is not required to exist; callers decide
        between skipping and failing.
    """
    env = os.environ.get("PLASTAX_CPP_DIR")
    if env:
        return pathlib.Path(env).expanduser().resolve()
    for name in ("plastax-cpp", "plastix"):
        if (_REPO.parent / name).is_dir():
            return _REPO.parent / name
    return _REPO.parent / "plastax-cpp"
