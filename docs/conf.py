"""Sphinx configuration for plastax (scaffold; full docs are Phase 3)."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

project = "plastax"
author = "Ayrton Chilibeck"
project_copyright = "2026, Ayrton Chilibeck"

try:
    release = _pkg_version("plastax")
except PackageNotFoundError:  # building docs without an installed package
    release = "0.0.0+unknown"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "sphinx_copybutton",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
# PEP 695 type parameters are documented under a custom "Type Args:" section
# (TOOLING.md docstring conventions).
napoleon_custom_sections = [("Type Args", "params_style")]

# Types render from signatures only; docstrings never duplicate them
# (pydoclint contract: --arg-type-hints-in-docstring=False).
autodoc_typehints = "signature"
autodoc_member_order = "groupwise"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

html_theme = "furo"

exclude_patterns = ["_build", "internal", "parallel_mnist_plan.md"]
