# plastax

Declarative plastic-network traits for JAX. A `plastax.Network` subclass
declares, in one place, the forward/backward passes, unit- and
connection-update rules, structural rules (connection growth and pruning, unit
addition and pruning), per-unit and per-edge SOA fields, and propagation model
of a dynamically structured network; the library assembles and jit-specializes the
corresponding step function at trace time — the JAX analogue of plastax-cpp's
C++ template metaprogramming.

Documentation: https://plastax.readthedocs.io

```
pip install plastax              # CPU
pip install "plastax[cuda13]"    # NVIDIA GPU (or [cuda12]); [tpu] for TPU
```

v1 scope: pipeline and topological propagation; structural dynamics in place:
connection growth (`ScoreAddConn` over a candidate grid or shortlist, or
`ProposeAddConn` from per-unit, per-connection or global proposals, at a cost
that follows the churn), connection pruning (`PruneConn`), and unit addition
and pruning (`AddUnit`, `PruneUnit`); named-monoid combines; Scheme-A
multi-device connection sharding; donation-based in-place state.

Per-unit proposal growth (`ProposeAddConn` with its default
`proposer = "per_unit"`) is the default for networks with unit addition: every
live unit, a newly spawned one included, proposes `proposals_per_proposer`
edges per step, so growth scores num_units x P candidates rather than
num_units². Growth
is deterministic: candidates commit in one total order, and proposals draw
from a counter-based rng keyed by the network seed and step; conformance
goldens shared with plastax-cpp pin both libraries to the same result.

plastax is built for **streaming**: one sample per step, with structure
changing between steps. For mini-batch training or evaluation of feed-forward
(topological) nets, `make_step(net, static, batch_size=B)` runs B samples per
step against shared connections and reduces the connection update over the
batch (exactly, for the `plastax.optim` bundles); see its docstring.
Deferred: unit addition and pruning under Scheme-A sharding (they run on a
single device today), generic associative combines, jax.Ref arena,
hijax-based primitive surface.

The C++ semantics oracle is the plastax-cpp library
(`include/plastax/traits.hpp`, `dispatch_cpu.hpp` in
[Sir-NoChill/plastax-cpp](https://github.com/Sir-NoChill/plastax-cpp)),
found by tests through the `PLASTAX_CPP_DIR` environment variable
(default: a `plastax-cpp` checkout beside this repository).

Contributing: see the development pages of the documentation (toolchain,
commit conventions, releasing). Licensed under the terms in `LICENSE`.
