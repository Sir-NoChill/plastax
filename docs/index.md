# plastax

Declarative plastic-network traits for JAX. A `plastax.Network` subclass
declares, in one place, the forward/backward passes, connection-update rule,
per-unit and per-edge SOA fields, and propagation model of a dynamically
structured network; the library assembles and jit-specializes the
corresponding step function at trace time.

This site is a scaffold; the full documentation structure (installation,
quickstart, concepts, examples, per-module API reference) is tracked in
`DISTRIBUTION_PLAN.md`, Phase 3.

## Installation

Requires Python >= 3.12.

```
pip install plastax
```

The default install pulls the CPU wheel of JAX. For accelerators, install
the matching JAX distribution yourself (for instance `pip install
"jax[cuda12]"`); plastax is backend-agnostic pure Python.

```{toctree}
:maxdepth: 1
:hidden:

api
```
