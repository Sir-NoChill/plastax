# plastax

Declarative plastic-network traits for JAX. A `plastax.Network` subclass
declares, in one place, the forward/backward passes, connection-update rule,
per-unit and per-edge SOA fields, and propagation model of a dynamically
structured network; the library assembles and jit-specializes the
corresponding step function at trace time — the JAX analogue of the plastix
C++ template metaprogramming.

Documentation: https://plastax.readthedocs.io

```
pip install plastax              # CPU
pip install "plastax[cuda13]"    # NVIDIA GPU (or [cuda12]); [tpu] for TPU
```

v1 scope: pipeline and topological propagation, AddConn/PruneConn dynamics
(grid or proposal growth, in place, at a cost that follows the churn),
named-monoid combines, single device, donation-based in-place state.

plastax is built for **streaming**: one sample per step, with structure
changing between steps. For mini-batch training or evaluation of feed-forward
(topological) nets, `make_step(net, static, batch_size=B)` runs B samples per
step against shared connections and reduces the connection update over the
batch (exactly, for the `plastax.optim` bundles); see its docstring.
Deferred: AddUnit/PruneUnit, generic associative combines, jax.Ref arena,
hijax-based primitive surface, multi-device sharding.

The C++ semantics oracle is the plastix library (`include/plastix/traits.hpp`,
`dispatch_cpu.hpp`).

Contributing: see the development pages of the documentation (toolchain,
commit conventions, releasing). Licensed under the terms in `LICENSE`.
