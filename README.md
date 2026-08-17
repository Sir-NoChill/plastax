# plastax

Declarative plastic-network traits for JAX. A `plastax.Network` subclass
declares, in one place, the forward/backward passes, connection-update rule,
per-unit and per-edge SOA fields, and propagation model of a dynamically
structured network; the library assembles and jit-specializes the
corresponding step function at trace time — the JAX analogue of the plastix
C++ template metaprogramming.

Design documents (read in order):

1. `../plastix-jax-lowering-analysis.md` — the rung ladder (trace-time
   metaprogramming -> composites + HLO transform -> Pallas -> native FFI)
   and how plastax fits the OpenXLA extension surface.
2. `../plastix-jax-rung0-design.md` — the rung 0 design this package
   implements: state representation, SoA-per-level arenas, retrace protocol,
   donation, monoid combine contract.
3. `IMPLEMENTATION_PLAN.md` — milestone plan and acceptance criteria for the
   initial implementation.

v1 scope: pipeline and topological propagation, AddConn/PruneConn dynamics,
named-monoid combines, single device, donation-based in-place state.
Deferred: AddUnit/PruneUnit, generic associative combines, jax.Ref arena,
hijax-based primitive surface, multi-device sharding.

References: `plastax.bib` alongside the design docs; C++ semantics oracle is
the plastix repo at `../plastix` (`include/plastix/traits.hpp`,
`dispatch_cpu.hpp`).
