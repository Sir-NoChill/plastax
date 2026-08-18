# plastax test suite layout

Mirrors the C++ one-policy-per-traits oracle structure (tests/test_plastix.cpp
et al.). Required modules, mapped to plan milestones (IMPLEMENTATION_PLAN.md):

- `test_pytree.py` (M1): NetworkState flatten/unflatten roundtrip;
  NetworkStatic meta fields hash/eq; changing a meta field changes the
  PyTreeDef; changing a leaf does not.
- `test_builder.py` (M1): builder -> finalize invariants (levels correct,
  buckets sorted by (dead, to_id), capacities obey capacity_policy).
- `test_topology.py` (M1): dense edge count = n_in*n_out; conv2d edge
  enumeration matches lax.conv_general_dilated shape semantics (positions,
  receptive fields, stride); initializer statistics sane; sequential id
  offsetting; from_topology equals the equivalent manual builder calls.
- `test_forward_pipeline.py` (M2): flat sweep vs numpy reference; one-hop
  latency semantics (dispatch_cpu.hpp:202-223); dead-slot null-scatter.
- `test_forward_topo.py` (M3): level walk vs numpy reference; equivalence
  with pipeline mode on a layered net after L steps.
- `test_backward.py` (M3): direction reversal (accumulate into source).
- `test_phases_elision.py` (M2): absent phases produce identical jaxprs to
  a hand-assembled subset (compare jax.make_jaxpr output structure).
- `test_update_conn.py` (M3b): UpdateConn incoming/outgoing two-pass
  ordering (a combined-in-one-pass implementation would fail); dst/src
  endpoint-binding convention for each direction; dead conns never
  updated; every bucket swept in topological mode; wired into
  build_phases via make_step.
- `test_mlp_xor.py` (M3b): the milestone's end-to-end acceptance --
  examples/mlp_xor.py's XorNet (forward+loss+backward+update_conn, SGD)
  trains the 4 XOR patterns to a low loss and classifies all 4 correctly;
  deterministic for a fixed seed.
- `test_update_prune.py` (M4): PruneConn tombstoning; derived live counts.
- `test_add_conn.py` (M4): K-bounded candidates, top_k selection,
  prefix-sum slot claim, overflow flag, level-preserving adds do not set
  needs_resort.
- `test_resort.py` (M4): recompute_levels vs host Kahn; resort produces
  sorted compacted buckets; retrace count == 1 per resort via
  jax.test_util.assert_num_jit_and_pmap_compilations; pure add/prune
  workload compiles exactly once.
- `test_ipc_multilayer.py` (M5): streaming-iPC acceptance --
  examples/ipc_multilayer.py beats the predict-previous baseline (2x
  margin), is seed-deterministic, and uses Pipeline propagation.
- `test_donation.py` (M5): donate_argnums=0 contract -- every input state
  buffer is deleted after the call (donation happens), pytree structure and
  every leaf shape/dtype survive it, and chaining the donated output into
  the next step raises nothing under a filter promoting the donation-failure
  warning to an error.
- `test_oracle_cpp.py` (M5): bit-level C++ parity via the deterministic
  manual-fcc example (fixed weights, tanh forward, fixed inputs), reproduced
  with from_topology + constant initialisers and driven through the host
  Driver; output activations pinned to golden values from the native binary
  at rtol=1e-4. (The PRNG-seeded flagship examples cannot match std::mt19937
  bit-for-bit; their parity is aggregate, in their own acceptance tests.)
