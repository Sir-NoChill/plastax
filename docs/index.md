# plastax

Declarative plastic-network traits for JAX. A `plastax.Network` subclass
declares, in one place, the forward/backward passes, unit- and
connection-update rules, structural rules (connection growth and pruning, unit
addition and pruning), per-unit and per-edge SOA fields, and propagation model
of a dynamically structured network; the library assembles and jit-specializes the
corresponding step function at trace time.

This site is a scaffold; the full documentation (quickstart, concepts,
examples, per-module API reference) is in progress.

## Installation

```
pip install plastax              # CPU
pip install "plastax[cuda13]"    # NVIDIA GPU (or [cuda12] / [gpu]); [tpu] for TPU
```

See {doc}`installation` for the accelerator extras and why they exist.

```{toctree}
:maxdepth: 1
:hidden:

installation
optimizers
api
changelog
```

```{toctree}
:maxdepth: 1
:caption: Development
:hidden:

development/tooling
development/tags
development/scopes
development/releasing
```
