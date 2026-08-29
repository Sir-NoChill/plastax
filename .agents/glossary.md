# plastax glossary

The vocabulary used across the codebase and docs. Terms are grouped; within a
group, ordered roughly by how fundamental they are.

## Representation

- **SoA (struct-of-arrays) arena** — the network's data stored as one array per
  field (`Columns = dict[str, Array]`), not one struct per element. Units are a
  single `Columns`; connections are a tuple of `Columns`, one per level bucket.
- **Column** — a single field's array across all slots (e.g. `weight`,
  `opt/m`). Keyed by `FieldSpec.name`.
- **`FieldSpec[DT]`** — the descriptor of one column: `name`, numpy `dtype`,
  `default`. Frozen, hashable, generic over the scalar type `DT`. Built via
  `FieldSpec.float32/int32/boolean/field`.
- **Built-in columns** — framework-owned specs: `FROM_ID`, `TO_ID`, `DEAD`
  (conn); `WEIGHT` (conn); `ACTIVATION`, `LEVEL` (unit). Their names are
  reserved.
- **Unit / connection (edge)** — a node and a directed weighted edge. Indices
  are distinct NewTypes `UnitIdx` / `ConnIdx` (erased to `Int32` arrays at
  runtime; the separation is a static-typing discipline).
- **Slot** — a physical row in a column, live or dead.
- **Tombstone / DEAD** — a dead slot is marked `DEAD=True` rather than removed;
  fresh slots default dead. Deletion = tombstoning. Live count is *derived* from
  the `DEAD` mask, never stored.
- **Bucket** — the per-source-level group of connection slots. TOPOLOGICAL mode
  has one bucket per source level (`level_capacities`); PIPELINE mode has
  exactly one flat bucket.
- **Capacity / headroom** — a bucket is over-allocated beyond its live edges so
  growth doesn't retrace every add. `capacity_policy` = `max(next_pow2(live),
  min_bucket)`.

## State & compilation

- **`NetworkStatic`** — the hashable static description (shapes, field layout,
  bucket capacities, I/O ids, propagation, sharding). The `jax.jit` cache key.
  Changes only on structural events.
- **`NetworkState[GS]`** — the mutable SoA pytree (`units`, `conns`, `globals_`,
  `needs_resort`). `GS` is the user's opaque globals pytree.
- **Retrace** — a re-trace + recompile of the step, triggered *only* by a new
  `NetworkStatic` (bucket growth or resort).
- **Donation** — `jax.jit(..., donate_argnums=0)`; the step consumes the input
  state buffers in place. Requires the step to be shape-preserving on state.
- **Monomorphization** — `make_step` specializing one `(Network, NetworkStatic)`
  pair into one cached jitted step (`step.py`).

## The step pipeline

- **Trait** — a declared policy slot on a `Network` subclass (`forward_pass`,
  `loss`, `backward_pass`, `update_conn`, `prune_conn`, `add_conn`,
  `reset_global`). Left `None` ⇒ elided.
- **Policy** — a pure per-element instance implementing a Protocol; assigned to
  a trait slot.
- **Phase** — a pure `state → (state, loss)` function `build_phases` produces
  from a present trait. Fixed order: forward → loss → backward → update_conn →
  prune_conn → add_conn → reset_global.
- **Phase elision** — an absent trait produces no phase and no jaxpr equations
  (Python-level, never `lax.cond`).
- **`StepInputs` / `StepResult`** — the jit input (`inputs`, optional `targets`)
  and output (`state`, `overflow`, `loss`).
- **View / write record** — `UnitView`/`ConnView` (read, by `(FieldSpec, Idx)`)
  and `UnitWrite`/`ConnWrite` (write, via `.of(...)`). The only sanctioned way a
  policy touches the arena.

## The sweep engine

- **Sweep** — the gather → vmapped map → `segment_reduce` → masked apply pass
  over one bucket.
- **Accumulate / apply split** — accumulate folds a bucket's edges into a
  carried per-unit accumulator; apply finalizes a unit once all feeding buckets
  are in. Pipeline uses a one-shot sweep; topological loops accumulate/apply
  across buckets.
- **Null-slot trick** — redirect a dead edge's target index to `num_units` (out
  of range) so `segment_reduce(mode=FILL_OR_DROP)` drops it — no shape-changing
  gather.
- **Monoid** — an associative combine + identity, lowered to `jax.ops.segment_*`.
  Named only in v1: `sum_`, `prod`, `max_`, `min_`. Generic `(op, identity)`
  raises `UnsupportedMonoidError`.
- **MonoidTree** — a pytree of monoids matching a struct-valued accumulator ("a
  product of monoids is a monoid"). A `ForwardPass.combine`.
- **`collective`** — the cross-shard all-reduce (`psum`/`pmax`/`pmin`) used under
  Scheme-A sharding (`prod` unsupported — no direct JAX collective).

## Structure dynamics

- **Level** — a unit's topological-sort depth (longest path from inputs).
  Inputs are level 0.
- **`initial_levels` / `recompute_levels`** — host-side Kahn/longest-path at
  build time; on-device bounded Bellman-Ford relaxation after structural change.
- **Resort** — rebucket all edges by (recomputed) source level after a
  non-level-preserving change; produces a new `NetworkStatic` → retrace.
- **`needs_resort`** — a device flag set only by the add_conn phase when it
  commits a non-level-preserving edge; checked host-side by the driver.
- **Overflow** — an add_conn candidate that was selected but found no free slot
  in its bucket; the driver grows the bucket and retries.
- **Retrace protocol** — the driver's host loop: run step; on overflow grow +
  retrace + retry; on `needs_resort` resort + retrace.

## Dynamic sparse training (DST)

- **SET** (Sparse Evolutionary Training, Mocanu 2018) — magnitude prune +
  **random** regrow. Random growth needs no gradient over absent edges.
- **RigL** (Evci 2020) — magnitude prune + **gradient** regrow; the absent-edge
  gradient is the delta-rule factorization `|grad_field[dst]·activation[src]|`, a
  local read. SET and RigL differ in exactly the `score` function.
- **Delta rule** — the per-edge weight gradient `dL/dw =
  grad_field[dst] · ACTIVATION[src]`, exact for any weighted-sum layer (dense or
  unrolled conv). Every optimizer and RigL's growth use it.
- **Optimizer bundle** — an `Optimizer` = `UpdateConn` policy + per-connection
  `state_fields` (`opt/…` columns, default 0) + `needs_step_counter`. State
  lives as SoA columns, so it shards for free and auto-zeroes on regrow.
- **Churn step** — a prune+grow rewiring step (distinct from a train step) that
  holds the live-edge count ~constant. See `examples/dst_sparse.py`.
- **Mask-based DST** — the standard baseline: dense weights + a binary mask,
  dense matmuls, `O(N²)` weights/grads/optimizer-state. The thing plastax's
  `O(E)` arena is claimed to beat at extreme sparsity.

## Sharding

- **Scheme A** — connections sharded edge-wise across a device mesh, units and
  globals replicated; monoid `collective` all-reduces partials. Implemented
  (`step._shard_map_step`, tested in `test_sharding.py`).
- **Scheme B** — level-band pipeline partitioning; `shard.balanced_level_cut`
  computes the min-max contiguous band boundaries (host-side numpy DP only).

## Propagation modes

- **`Propagation.TOPOLOGICAL`** — level-ordered sweep, one bucket per source
  level; cycles rejected at build. The default.
- **`Propagation.PIPELINE`** — one synchronous flat sweep over the previous
  step's activations; cycles allowed (recurrent nets, echo-state reservoirs);
  levels cosmetic.
