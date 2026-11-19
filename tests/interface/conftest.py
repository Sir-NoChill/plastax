"""Configuration for the interface tests."""

import pathlib

import pytest

HERE = pathlib.Path(__file__).resolve().parent


def pytest_configure(config: pytest.Config) -> None:
    """Register the marker for tests of behaviour the design has not settled."""
    config.addinivalue_line(
        "markers",
        "open_question(reason): tests behaviour the design document leaves undecided; "
        "written against a best guess",
    )


def pytest_ignore_collect(
    collection_path: pathlib.Path, config: pytest.Config
) -> bool | None:
    """Collect these tests only when a run names this directory or a file in it.

    The interface they test does not exist yet, so most of these files fail at import. A
    run of the rest of the suite (`pytest`, `pytest tests`) must not be stopped by that.
    Remove this hook once the interface is in place.
    """
    root = config.invocation_params.dir
    for arg in config.args:
        named = (root / arg.split("::")[0]).resolve()
        if named == HERE or HERE in named.parents:
            return None
    return True
