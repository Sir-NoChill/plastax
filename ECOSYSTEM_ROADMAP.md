# plastax ecosystem roadmap (internal)

Internal planning doc: what an end user needs, beyond the v1 core, to build
algorithms *within the plastax paradigm* -- highly sparse, dynamically
structured networks expressed as trait bundles over SoA arenas. Companion to
`DISTRIBUTION_PLAN.md` (distribution) and `IMPLEMENTATION_PLAN.md` (core).
Citations in `distribution-plan.bib`. Timeline anchor: publication target
mid-October 2026; v0.1.0 on PyPI is the artifact the paper cites.

Standing constraints (do not relitigate here; see rung0 decision log):
SoA-per-level arenas, donation, named monoids only in v1, AddUnit/PruneUnit
out of v1 scope, conv unrolled per-edge, continual/streaming focus (static
matmul parity a non-goal).

Everything below ships in-tree (locked 2026-08-21): `plastax.optim`,
`plastax.heuristics`, `plastax.tools`. optax and friends appear only as
test oracles, never runtime deps.

---

## Track A -- plastax.optim (v0.1.x, before final paper benchmarks)

The `optim` stub already fixes the shape: an Optimizer is a bundle --
UpdateConn policy + per-connection `state_fields` + optional step counter in
globals. Because optimizer state lives as SOA columns, it shards with the
connections under scheme-A for free; that is a selling point worth a docs
page and a paper sentence.

- A1 Settle the two open contracts blocking promotion from the
  parallel_mnist AdamUpdateConn prototype:
  - field declaration: how a Network merges `optimizer.state_fields` into
    `extra_conn_fields` without name collisions (namespace prefix, e.g.
    `opt/m`, `opt/v`);
  - step counter: globals slot reserved iff `needs_step_counter`, bumped by
    the assembled step, reset by ResetGlobal.
- A2 Implementations, in order: `sgd` (stateless baseline), `momentum`,
  `adam`, `adamw`, `rmsprop`. Each validated against optax on dense
  topologies to fp tolerance [optax][deepmind-jax].
- A3 Sparse-aware semantics: on AddConn, new connections need initialized
  optimizer state (zeros for moments) -- specify this as part of the
  Optimizer protocol (an `init_fields` per-connection default), since
  RigL-style regrowth explicitly zeroes moments for regrown weights
  [rigl].
- A4 Acceptance: mlp_xor and parallel_mnist examples switch to
  `plastax.optim`; optax-oracle tests marked `slow`; docs page "Optimizers
  as trait bundles".

## Track B -- plastax.heuristics: growth/prune defaults (v0.2)

End users implementing dynamic sparse training should compose known-good
policies instead of hand-writing AddConn/PruneConn. Ship the literature
defaults as named, parameterized trait implementations:

- B1 Prune policies (PruneConn):
  - `magnitude(fraction | threshold)` -- the standard baseline [sparsity-survey];
  - `set_prune(zeta)` -- smallest-positive + largest-negative fraction, per
    SET [set];
  - schedule support: constant, cosine decay of the update fraction as in
    RigL [rigl].
- B2 Growth policies (AddConn), building on the existing candidate-unit
  shortlist mechanism:
  - `random_regrow(k)` -- SET-style random regrowth [set];
  - `hebbian_covariance(k)` -- score candidates by pre/post activity
    correlation accumulated in existing per-unit fields;
  - gradient-magnitude growth a la RigL needs dense-gradient sampling over
    absent connections -- that conflicts with the SoA-arena design for large
    nets; scope it to shortlist-restricted gradient scoring and note the
    deviation from the paper's dense scoring [rigl].
- B3 Rewiring schedules: pair a prune and a growth policy with a cadence
  (every N steps, fraction schedule) into one composable `Rewiring` bundle;
  DEEP R's walk-in-parameter-space framing is the reference for
  prune+regrow-as-one-operator [deepr].
- B4 Acceptance: a churn benchmark (`plastix-bench/12_churn`) variant runs
  purely on `plastax.heuristics` defaults; each policy has an oracle test
  against a numpy reference implementation.

## Track C -- trait-author toolkit: plastax.tools (v0.2-v0.3)

What a user needs to *develop and debug* a new algorithm in the paradigm:

- C1 Introspection: public API over the Driver's retrace/overflow counters
  (currently test-internal); per-phase timing via
  `jax.profiler`-compatible annotations; a `state_summary(state, static)`
  report (occupancy per level, dead fraction, weight stats).
- C2 Topology export: `to_networkx(static, state)` and a graphviz dot
  emitter for small nets -- indispensable for debugging growth heuristics.
- C3 Testing utilities: generalize the C++ oracle harness pattern
  (`test_oracle_cpp.py`) into a documented recipe; export the
  jaxtyping+beartype pytest wiring as a copy-paste snippet for downstream
  suites.
- C4 A documented "write your own trait" tutorial: implement a leaky
  integrate-and-fire unit + STDP-flavored UpdateConn end to end. This is
  the page that defines the paradigm for outsiders; budget real time for
  it.
- C5 Acceptance: a downstream user can implement and validate a new
  UpdateConn without reading plastax internals -- measured by the tutorial
  being self-contained.

## Track D -- core capabilities the ecosystem will demand (v0.3+, post-paper)

Deferred core items, ordered by how many ecosystem doors they open:

- D1 AddUnit/PruneUnit: prerequisite for constructive algorithms
  (cascade-correlation and descendants [cascor]) and true neurogenesis-style
  models. Largest arena-design impact (per-level occupancy changes);
  deliberately post-paper.
- D2 Generic associative monoids: unblocks user-defined combine semantics
  (e.g. max-plus for shortest-path-like propagation) beyond the named set.
- D3 jax.Ref arenas / hijax surface: revisit per the rung ladder once
  upstream stabilizes; affects donation contract, so gate on a major
  version bump.
- D4 Sharding beyond scheme-A: driven by whatever the parallel_mnist and
  churn-scaling results demand, not speculatively.

## Sequencing against the paper

1. Now -> submission: DISTRIBUTION_PLAN phases 0-4 (installable artifact,
   docs), Track A (benchmarks should use plastax.optim, not example-local
   optimizers).
2. Submission -> camera-ready: Track B (heuristics strengthen the
   continual/streaming story), C1-C2.
3. Post-paper: C3-C5, Track D.

## Deviations

(Date + rationale, as elsewhere.)
