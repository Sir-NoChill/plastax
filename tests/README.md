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
- `test_update_prune.py` (M4): UpdateConn incoming/outgoing two-pass
  ordering; PruneConn tombstoning; derived live counts.
- `test_add_conn.py` (M4): K-bounded candidates, top_k selection,
  prefix-sum slot claim, overflow flag, level-preserving adds do not set
  needs_resort.
- `test_resort.py` (M4): recompute_levels vs host Kahn; resort produces
  sorted compacted buckets; retrace count == 1 per resort via
  jax.test_util.assert_num_jit_and_pmap_compilations; pure add/prune
  workload compiles exactly once.
- `test_donation.py` (M5): donation warning promoted to error (pytest
  filterwarnings); step is shape-preserving on every leaf.
- `test_oracle_cpp.py` (M5): golden-file parity vs the C++ examples
  (tolerance-based; see plan section "Oracle harness").
