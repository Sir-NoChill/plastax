"""Locate the plastax-cpp checkout (the C++ implementation).

Cross-library artifacts -- the RNG golden, the conformance vectors -- live in
the plastax-cpp tree. Its location comes from the ``PLASTAX_CPP_DIR``
environment variable, defaulting to a ``plastax-cpp`` directory beside this
repository. Tests that need the checkout skip when it is absent; scripts fail
loudly instead, since silently emitting goldens nowhere helps nobody.
"""

from __future__ import annotations

import os
import pathlib

_REPO = pathlib.Path(__file__).resolve().parents[1]


def plastax_cpp_dir() -> pathlib.Path:
    """Path to the plastax-cpp checkout.

    Returns:
        ``$PLASTAX_CPP_DIR`` if set, else ``../plastax-cpp`` relative to this
        repository's parent. The path is not required to exist; callers decide
        between skipping and failing.
    """
    env = os.environ.get("PLASTAX_CPP_DIR")
    if env:
        return pathlib.Path(env).expanduser().resolve()
    return _REPO.parent / "plastax-cpp"
