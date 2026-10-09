# plastax architecture

The layout of the project, what each module owns, how a step function is
assembled at trace time, and **where a given contribution goes**. This is the
reference the `plastax-architecture` skill loads. Read
[`invariants.md`](invariants.md) alongside it — the invariants constrain
everything below. Vocabulary is in [`glossary.md`](glossary.md).

---

## 1. The dependency stack

plastax is layered. Lower layers never import upper ones.

```
                    driver.py        (host loop: retrace / overflow / resort)
                        │
   builder.py           step.py      (assemble + jit + donate + shard_map)
      │                   │
      └───────┬───────────┤
              │        phases.py      (traits ──▶ ordered pure phases; elision)
              │           │
   topology.py│        sweep.py       (gather→vmap map→segment_reduce→apply)
   shard.py   │        topo.py        (levels, resort, capacity)
   (leaves)   │           │
              └───────────┤
                     traits.py        (Network base + policy Protocols)
                        │
        views.py ── state.py ── monoid.py
                        │
                    _types.py         (leaf: NewTypes, FieldSpec, enums)
```

- **`_types.py`** is the graph leaf: index NewTypes, `FieldSpec`, built-in
  columns, `Propagation`, `ShardSpec`. No plastax imports, no JAX compute.
- **`monoid.py`** is pure algebra with no dependency on the arena at all —
  independently testable.
- **`state.py`** depends only on `_types` (and, lazily inside `grow_bucket`, on
  `topo.capacity_policy` via a *local* import to break a cycle —
  `state.py:161`; keep it local).
- **`sweep.py`** is the integration point: it consumes `_types`, `state`,
  `views`, `monoid`, and the `ForwardPass`/`BackwardPass` Protocols from
  `traits`.
- **`topology.py`** and **`shard.py`** are pure host-side (numpy) leaves — no
  other plastax imports.

---

## 2. The two-tier state (this is the whole design)

Everything hinges on splitting the network into a **static** part and a
**dynamic** part:

- **`NetworkStatic`** (`state.py:22`) — a frozen, `register_dataclass`
  dataclass whose every field is `static=True`: `num_units`, `propagation`,
  `unit_fields`/`conn_fields` (tuples of `FieldSpec`), `level_capacities`
  (one bucket per source level; a 1-tuple for PIPELINE), `input_ids`,
  `output_ids`, `sharding`. It is **hashable** and is the `jax.jit` cache key.
  It changes **only** on structural events (bucket growth, resort).
- **`NetworkState[GS]`** (`state.py:62`) — the mutable SoA pytree:
  `units: Columns`, `conns: tuple[Columns, ...]` (one dict of `(capacity,)`
  arrays per bucket), `globals_: GS` (opaque user pytree), `needs_resort` (a
  device bool checked host-side between steps). `Columns = dict[str, Array]`,
  one array per `FieldSpec.name`.

Consequence: mutating leaf *values* never changes the cache key, so the jitted
step is reused. Only `grow_bucket`/`resort` mint a new `NetworkStatic` and
force a retrace. This is invariant #2.

Fresh connection slots default `DEAD=True` (`_types.py:120`) — allocation is
tombstone-first; a slot becomes live only when explicitly written. Live counts
are always derived (`live_conn_count`, `state.py:119`), never stored
(invariant #3).

---

## 3. The SoA arena and how policies touch it

Per-unit and per-edge data live as **columns** (one array per field), bucketed
by source level for connections. User policy code **never indexes columns
directly**. Instead:

- Reads go through `UnitView`/`ConnView` (`views.py`), indexed by a
  `(FieldSpec, UnitIdx|ConnIdx)` tuple → a scalar.
- Writes are returned as `UnitWrite`/`ConnWrite` records (`views.py`), built
  with `.of((spec, value), …)`.

`UnitWrite`/`ConnWrite` are **deliberately not pytree-registered**: `sweep.py`
unwraps `.fields` to a plain dict before `vmap`, so the wrapper is never
batched. Do not register them (invariant #6, and see `sweep.py:209`).

Policies are **pure per-element functions** run under `vmap` over an entire
bucket (or all units). They cannot see other elements' state or raw columns.
This is what makes them shardable and jit-friendly.

---

## 4. The sweep engine (`sweep.py`)

One bucket at a time: **gather → vmapped map → segment_reduce → masked apply**.

- **Null-slot trick** (invariant #4): before `segment_reduce`
  (`mode=FILL_OR_DROP`), a dead edge's target index is redirected to
  `num_units` (out of range) so its contribution is silently dropped — no
  shape-changing masked gather.
- **Accumulate/apply split.** `build_forward_sweep`/`build_backward_sweep`
  (`sweep.py:280`, `:395`) are one-shot (identity-in, finalize-all) — correct
  only for a single-bucket **pipeline** sweep. **Topological** mode instead
  composes `build_forward_accumulate`/`build_forward_apply` (and backward
  counterparts) across a bucket loop in `phases.py`, carrying a per-unit
  accumulator so a unit finalizes only after every bucket that can feed it
  (skip connections may live in any earlier bucket).
- **Direction pairing.** Forward accumulates into `TO_ID` (destination);
  backward into `FROM_ID` (source).
- **Conn updates** (`build_incoming_conn_update`/`build_outgoing_conn_update`,
  `sweep.py:528`, `:546`) write only the edge's own row, so they use a plain
  `where(dead, old, written)` merge — no segment reduction needed.

Accumulator pytree structure must exactly match the `MonoidTree` (`combine`) —
`tree_map` zips them leaf-by-leaf ("a product of monoids is a monoid").

---

## 5. Trait declaration → phase assembly → jitted step

### Declaration (`traits.py`)

A `Network[GS]` subclass declares its algorithm as **class attributes holding
policy instances** — no methods to override:

```python
class Net(px.Network[None]):
    forward_pass = SigmoidForward()      # required
    backward_pass = SigmoidBackward()    # any of these may be omitted (→ None)
    loss = MSELoss()
    update_conn = px.optim.adam(1e-3, Delta).update_conn()
    extra_conn_fields = px.optim.adam(1e-3, Delta).state_fields
    propagation = px.Propagation.TOPOLOGICAL
```

`__init_subclass__` (`traits.py:362`) runs `_validate_traits` **once at
class-definition time**: `forward_pass` must be present; every configured slot
is checked against its `@runtime_checkable` Protocol (method-name presence, not
signatures); `combine` MonoidTrees and extra field names are validated (no
collision with reserved builtin columns).

The policy Protocols:

| Protocol | Key methods | Accumulates into |
|---|---|---|
| `ForwardPass[Acc,GS]` | `map(u,dst,src,c,cid,g)→Acc`, `apply(u,i,g,acc)→UnitWrite`; attr `combine:MonoidTree` | destination unit |
| `BackwardPass[Acc,GS]` | same shape | source unit |
| `Loss[GS]` | `calculate_loss(u,outputs,targets,g)→(scalar, seed)`; attr `seed_field:FieldSpec` | the seed field, at the output units |
| `UpdateUnit[GS]` | `update(u,i,g)→UnitWrite` | every live unit |
| `UpdateConn[GS]` | `incoming(...)→ConnWrite`, `outgoing(...)→ConnWrite` | the edge (two-pass) |
| `PruneConn[GS]` | `predicate(u,c,cid,g)→Bool` | tombstones edges |
| `ScoreAddConn[GS]` | `score(u,src,dst,g)→Float`, `init(u,src,dst,g)→ConnWrite`; optional `importance(u,i,g)→Float` (shortlists) | grows edges (scored pairs) |
| `ProposeAddConn[GS]` | attr `proposals_per_proposer:int`; `propose(...)→(src,dst,score)` per `proposer`, `init(...)→ConnWrite` | grows edges (proposals) |
| `ResetGlobal[GS]` | `reset(g)→GS` | globals, between episodes |

Both rule kinds carry the §10 knobs *structurally* (via `getattr`, validated
in `_validate_traits`): `selection` / `max_new_per_level` / `max_new_per_step`
/ `threshold(g)`, the window (`max_level_gap`, `direction`,
`allow_self_loops`), `dedupe_live` / `dedupe_step` (both default False),
`trigger` and `on_overflow`. A `ScoreAddConn` also picks `candidates`
(`exhaustive`, `shortlist`, `shortlist_per_level`; the shortlists take
`shortlist_size` + `importance`). `add_conn` must satisfy exactly one of the
two Protocols; `predicate_add_conn` adapts a boolean predicate to a
`ScoreAddConn`. Growth reports `grown` and `overflow` on the state, and
`Network.structural_interval` gates the structural phases to every n-th step.

`Network.unit_capacity` (default None) sizes the unit columns to a fixed slot
count and adds the `PRUNED` column (free slots marked); `_apply_masked`, the
loss's seed write, the fused prune's forwarded fields and growth's candidate
validity skip any slot `state.live_unit_mask` excludes; a loss policy reads
the same mask through `UnitView.live` to leave such an output out of the loss.
`Network.max_levels` (default 1024) bounds the unit levels unit addition may
assign.

### The loss contract

The loss is **whole-output**: `calculate_loss(u, outputs, targets, g)` runs
once per step (per sample when batched) over every output unit -- `outputs`
are the output ids in builder order, aligned with `targets`, read through
`UnitView.gather` -- and returns the scalar loss plus the `(num_outputs,)`
gradient seed dL/d(output). The framework writes the seed into the policy's
declared `seed_field` (a float unit column of the network) at the output ids
and nothing else; the scalar becomes the phase's contribution to
`StepResult.loss` (the batch mean when batched). Because one call sees every
output, losses that couple the outputs are expressible; `SoftmaxCrossEntropyLoss`
ships in `traits.py` (max-subtracted log-sum-exp, operation-for-operation the
same float32 arithmetic as plastax-cpp's, pinned by the `loss_v1` goldens).

The backward accumulator is **read-only** to every policy: it exists only as
the value the backward walk carries, and reaches a rule as
`BackwardPass.apply`'s `acc` argument. The loss cannot write it; a backward
pass picks the seed up from the seed field instead (the output level's own
`acc` is the identity, since no edge sources from the deepest level). In
plastax-cpp the same contract holds: `CalculateLoss` returns the scalar
(`GetLastLoss()`), writes its declared `SeedField`, and `GetBackwardAcc` is a
const accessor; its built-in losses declare the accumulator itself as their
seed field, which is how that library hands the seed to the backward pass.

### Assembly (`phases.py`)

`build_phases(net, static, *, overflow_sink)` (`phases.py:64`) emits an ordered
tuple of pure `state → (state, loss_contribution)` phase functions in the
**fixed order**:

```
forward → loss → backward → update_unit → update_conn → prune_conn → add_conn → reset_global
```

Each phase is appended **iff its trait slot is not `None`** (forward is
unconditional). This is **phase elision** (invariant #1): an absent phase means
no equations in the jaxpr, verified by `test_phases_elision.py`. Forward and
backward branch on `net.propagation` (single flat sweep for PIPELINE; a
per-bucket level walk for TOPOLOGICAL). `build_add_conn_phase` (`phases.py:426`)
is the most complex: candidate grid (full or shortlisted) → level-window filter
+ dedup vs live edges → `score` → per-bucket `top_k` → prefix-sum free-slot
claim (`xla_claim`, or `triton_claim`'s three jax_triton kernels on NVIDIA,
picked by `make_step(growth=...)`) → commit only finite-scored candidates → set `needs_resort` if a
committed edge isn't level-preserving. A `-inf` score is a **hard veto**. Under
Scheme-A it is device-resident and shards byte-identically: the dedup all-
reduces (so every shard agrees on the candidate set and `top_k`), and the
free-slot claim runs over each shard's capacity slice with an all-gathered
global free-slot rank sending each new edge to the one shard that owns its slot
(`total_free` a `psum`, keeping `overflow`/`needs_resort` replicated).

### Monomorphization (`step.py`)

`make_step(net, static)` (`step.py:54`, cached on `(net, static)`):
scatters `StepInputs.inputs` onto `units[ACTIVATION]` at `input_ids` **before
any phase**, runs the phases in order summing `total_loss`, returns
`StepResult(state, overflow, loss)`, wraps in `_shard_map_step` iff
`static.sharding`, then `jax.jit(traced, donate_argnums=0)`. Donation donates
the whole state pytree, so the step **must be shape-preserving** on every leaf
(invariant #5).

> The `overflow_sink` is a length-1 Python list the add_conn phase mutates
> exactly once during the single trace; `step.py` reads it into
> `StepResult.overflow`. This is safe only because `jax.jit` traces the body
> once (`step.py:141`).

### Batched step (`step.py` `make_step(..., batch_size=B)`, `phases.build_batched_phases`)

Streaming (B = None) is the primary mode. With a batch size, the phases split
three ways: **per-sample** (forward, loss, backward, update_unit) vmapped over the batch
with conns/globals broadcast; the **update** reduced over the batch
(`build_batched_update_conn`: the exact `per_sample` + `incoming_batched` pair
if the UpdateConn declares it -- every `optim/` bundle does -- else the mean of
the per-sample writes; both accumulate in a `fori_loop`, O(capacity) memory);
and **structural** (prune, add, reset) run once on `batch_mean_units`. Unit
columns in the state stay `(num_units,)` and hold the batch mean. PIPELINE nets
are rejected. Measured on GPU the per-sample cost falls only ~2x from B = 1 to
128 on the edge-list layout (each pass touches every edge once per sample);
the CSR layout addresses that for linear passes:

- **CSR layout** (`make_step(..., layout="auto" | "edge_list" | "csr")`,
  `phases.build_csr_forward` / `build_csr_backward`): a pass declaring
  `linear_input` (map == `WEIGHT * u[F, other]`, combine == `sum_`, read by
  `phases.linear_input_field`) runs as one cuSPARSE sparse-dense product per
  bucket over the batch. The view is rebuilt on device every step from the
  arena (`bucket_csr`: one radix sort; dead slots as explicit zeros), so it
  never goes stale under in-place churn; under Scheme-A the per-shard partial
  products are all-reduced. cuSPARSE lowering is a jax config flag scoped to
  the step's calls (`step._with_cusparse`).
- **Triton layout** (`layout="triton"`, `phases.triton_bucket_product`): the
  same linear passes as one edge-once Triton kernel per bucket, called
  through `jax_triton` (the optional `plastax[triton]` extra; triton is
  imported lazily) -- each edge read once for the whole batch, relaxed
  atomics into the targets, no sort. Dead slots are masked in the kernel
  (a null target row serialises its atomics on one address), and the
  backward (targets FROM_ID, in runs in a source-major bucket) sums each run
  in a tile before one atomic for a padded batch of at most 8. NVIDIA GPUs only
  (`phases.nvidia_triton_available`); anywhere else, and under Scheme-A,
  "triton" runs `phases.xla_bucket_product`, the same edge-once product in
  plain XLA (how the CPU tests exercise the layout). It replaced a Pallas
  Triton kernel: that lowering is deprecated in jax, and Pallas' Mosaic GPU
  backend cannot express a scatter-add into arbitrary rows (a low-level
  `inline_mgpu` prototype ran 1.03-1.6x slower than Triton).
  `bucket_product(engine)` is the seam the layouts share with the level walks.
- **"auto"**: on an NVIDIA GPU, Triton for 2 <= B <= 64 (when jax_triton is
  installed) and CSR above; on every other backend (AMD GPU, TPU, CPU) the
  XLA edge-once product for B >= 2 (same speed as the per-sample edge list,
  about 2.6x smaller temporaries compiled for TPU); the edge list at B = 1
  and for every non-linear pass. The CSR step keeps jit's `.trace`/`.lower`
  (`step._CusparseStep`), so every layout AOT-compiles (`docs/development/tooling.md`, TPU).

### Prune fused into the forward (`make_step(fuse_prune=)`)

A streaming step with a forward and a prune_conn would read every bucket's
edge columns twice. `phases.plan_prune_fusion` decides once, at the step's
first trace (it needs the globals' shapes), whether the predicate may run
inside the forward sweep, by tracing the policies to jaxprs:

- the predicate's reads (plus DEAD) must not be written by loss, backward or
  update_conn;
- a unit field it reads that the forward's `apply` writes must depend only on
  the unit id, the globals and columns the forward does not write -- never
  on the accumulator. Its post-forward value is then computed up front
  ("forwarded"); any other forward-written read is not fusable.

A fused step runs `build_fused_forward_prune_phase` in place of the forward
and `build_prune_merge_phase` at the prune slot, so loss/backward/update_conn
still see the old dead mask. With an unsharded linear forward on an NVIDIA
GPU, each bucket is one Triton kernel (`triton_forward_prune`): gather,
relaxed atomic scatter-add, the predicate translated from its jaxpr by
`_PredicateTranslator` (exact integer / compare / select ops only), the
tombstones written in place, and the free-slot block counts that add_conn's
claim then reuses (`free_sink`): in `TRITON_CLAIM_BLOCK` (256-slot) blocks
straight into `triton_claim(block_counts=)` when the Triton claim runs
(`triton_claim_applies`), else in 1024-slot blocks for `xla_claim`. XLA cannot do this in one pass (a scatter is
never a multi-output fusion root), so `fuse_prune="auto"` keeps the two-pass
step everywhere else; `"xla"` forces the XLA-lowered fused step (the CPU
correctness reference, also valid under Scheme-A). Batched steps never fuse.
The decision is `step.prune_fusion.plan`.

### Host loop (`driver.py`)

`Driver.step(inputs)` (`driver.py:51`) runs the jitted step and reacts to the
flags it returns — the **retrace protocol**:

- **overflow** → for each full bucket, `state.grow_bucket` (new `NetworkStatic`
  → `make_step` retrace), retry the *same* inputs against the failed attempt's
  **output** state (the donated input buffers may be gone).
- **needs_resort** → `topo.resort` (new bucket layout, new `NetworkStatic`),
  rebuild the step, return.
- else commit.

`Driver(..., check_every=N)` with N > 1 reads the flags back only every N
steps (overflow OR-accumulated on device; `needs_resort` is sticky in state):
no retry of an overflowing step (buckets short of `max_new_per_level` free slots
grow at the check) and a resort deferred to the check. Opt-in, for launch-
bound small nets; N = 1 is the exact protocol above.

`topo.resort` (`topo.py:149`) recomputes levels (`recompute_levels`,
Bellman-Ford relaxation bounded by `kahn_max_depth`), redistributes edges into
new per-level buckets (prefix-sum compacting scatter + stable sort on
`dead*num_units + from_id` to restore the builder's source-major order -- live
edges first, grouped by source, for scatter-add performance; in-place churn
loosens it again, so no sweep relies on it), and sizes new capacities via `capacity_policy` with the build's
recorded headroom and alignment (`static.capacity_headroom` / `capacity_align`). It returns a
**new** `(static, state)` — the caller must retrace.

---

## 6. Where does my contribution go?

Use this table first. The overwhelmingly common case is the top row.

| I want to add… | Touch | Do NOT touch | Notes |
|---|---|---|---|
| A **learning rule / plasticity algorithm** (Hebbian, a new forward/backward, a loss, a prune or grow policy) | A **new policy class** implementing an existing Protocol, in an example or a user module | Any framework module | Assign it as a class attribute on a `Network` subclass; validation is automatic. This needs **no core change**. Use the `plastax-algorithm-scaffold` skill. |
| A **new optimizer** | `optim/_<name>.py` + register in `optim/__init__.py` | `traits.py`, `phases.py`, `step.py` | Implement the `Optimizer` bundle: `state_fields` (`opt/…` columns, default 0), `needs_step_counter`, `update_conn()`. Delta-rule gradient. See §7. |
| A **new topology generator** | `topology.py` | `builder.py`, `state.py` | Return a `Block` (`num_units` + `edges(key, offset_in, offset_out)→EdgeSet`). Host-side numpy only; sets *initial* weights only. |
| A **new named monoid** | `monoid.py` (`_Named`, the four reducer/identity/pairwise/collective tables) | anything arena-aware | Keep it pure algebra. Do not un-guard the generic `(op, identity)` path without real lowering. |
| A **new phase category** (beyond the seven) | `traits.py` (Protocol + slot) **then** `phases.py` (`_build_<name>_phase` + wire into `build_phases`, deciding order) **then** likely `sweep.py` helpers | `step.py`, `topo.py` | Rare. `step.py`/`topo.py` are generic over the phase list. |
| A **new propagation/scheduling strategy** (neither pipeline-flat nor topological-level-walk) | `phases.py` forward/backward branch, `_types.Propagation`, `topo.py` bucket-count derivation | — | Rare and invasive. |
| **Arena layout** change (a new built-in column, bucket shape) | `state.py` (+ `_types.py` for a shared column) | `sweep.py` algorithm logic | A `NetworkStatic` field change is a retrace/cache-key change. |
| A **new accessor space** | `views.py` | — | Keep views pure and dict-backed; do not pytree-register write records. |
| **Caching / jit / donation / Scheme-A sharding** mechanics | `step.py` | any per-algorithm logic | `step.py` must stay algorithm-agnostic. |
| **Host retrace/overflow/resort** policy | `driver.py` (control flow) or `topo.py` (level/capacity math) | device numerics | driver only *calls* `grow_bucket`/`resort`; it computes no capacities. |
| **Scheme-B partition** math | `shard.py` | any JAX-traced code | Pure numpy DP; runs at build/resort time, never per-step. |

Per-module "add here / not here" detail is embedded in each module's docstring;
the routing above is the summary.

---

## 7. The optimizer bundle contract (`optim/`)

An optimizer is **not** a special object — it is a trait bundle
(`docs/optimizers.md`). The `Optimizer` Protocol (`optim/__init__.py:47`):

- `state_fields: tuple[FieldSpec[np.generic], ...]` — extra per-connection
  columns, namespaced `opt/…` (e.g. `opt/m`, `opt/v`, `opt/t`). Each
  `FieldSpec.default` **is** the regrow-init: on growth the framework
  resets an edge's untouched fields to their defaults, so a stateful
  optimizer's moments start at zero on a regrown edge — exactly RigL/SET's
  "zero the moments for regrown weights", with no work from the growth policy. A
  growth policy writes `WEIGHT` (+ its own fields) only, never `opt/…`.
- `needs_step_counter: bool` — all shipped optimizers keep this **False** (any
  step count is a per-edge `opt/t` column, not a global). Setting it True is the
  untrodden globals path — flag it.
- `update_conn() → UpdateConn` — builds the policy.

Every optimizer forms the per-edge gradient by the **delta rule**
`dL/dw = grad_field[dst] * ACTIVATION[src]` (exact for any weighted-sum layer,
dense or unrolled conv), reads/writes its `opt/…` columns, and returns a
`ConnWrite`. `outgoing` is a no-op for all shipped optimizers. `g` is typed
`object` so one instance satisfies `UpdateConn[GS]` for any `GS` — never read
`g`'s fields. This relies on phase order `forward → loss → backward →
update_conn`. To add one, follow §6 and the `plastax-algorithm-scaffold` skill.

---

## 8. Testing conventions

The contract:

- **Location/naming:** flat `tests/test_<topic>.py`, one file per
  trait/mechanism. Non-test helper scripts (e.g. subprocess bodies) drop the
  `test_` prefix (`tests/sharding_equiv.py`).
- **Determinism:** `tests/conftest.py` forces `JAX_PLATFORMS=cpu` and fakes 4
  CPU devices **before JAX imports**. Do not override. Seed every RNG.
- **jaxtyping runtime checks:** `addopts` wires
  `--jaxtyping-packages=plastax,beartype.beartype`, so every annotated
  signature is a runtime contract during tests. `shard_map` is incompatible
  with this layer — run such checks in a subprocess (`test_sharding.py` →
  `sharding_equiv.py`).
- **Donation contract:** `filterwarnings=["error:.*Some donated buffers were
  not usable.*"]` turns donation waste into a test failure globally.
- **Retrace-count contract:** use
  `jax._src.test_util.assert_num_jit_and_pmap_compilations` (see
  `test_resort.py`); keep eager construction outside the counted block.
- **Oracle tolerances** — pick by comparison type:
  - external oracle (optax, C++ binary): `rtol=1e-4, atol=1e-5`
  - internal numpy reference, identical reduction order: `rtol=1e-6, atol=1e-6`
  - cross-mode equivalence (pipeline vs topological): `rtol=1e-5, atol=1e-5`
  - exact invariants (regrown state zeroed): `atol=0.0`
  Document *why* a tolerance was chosen, in the existing files' style.
- **Slow marker:** `@pytest.mark.slow` for optax/heavy oracles (excluded from
  the pre-push fast suite). Use `pytest.importorskip` for optional deps.
- **Example-backed acceptance:** `examples/` is not on `sys.path`; tests load an
  example by file path (`importlib.util.spec_from_file_location`; see the
  `_load_example` helper). A good example's `main()` asserts its own success
  criteria so it doubles as its acceptance test.

## 9. What makes a good example (`examples/`)

Flat `examples/<name>.py`, runnable (`if __name__ == "__main__": main()`), with
a run-command line in the docstring. Each demonstrates **one conceptual point**
stated up front (e.g. "SET and RigL differ in exactly one expression"; "a
convnet is just a different topology"). Reuse existing trait definitions
(`SigmoidForward`/`SigmoidBackward`/`MSELoss`/`GradPreAct`/`LossGrad` live in
`mlp_xor.py`). An example must use only **public** traits — if it needs an
internal API, that is a signal the public surface is missing something. Cross-
reference the C++ oracle file when porting.

## 10. Growth: the target model (design record)

The growth rework lands in stages; this section is the normative design the
stages implement; as of G5 the code implements all of it except the unit
lifecycle that `on_units_added` reads (`NetworkState.units_added` stays 0
until it lands). Where code disagrees with this section, the code loses.

Two strategies, one deterministic selection pipeline:

- **Propose** (default): a proposer emits `proposals_per_proposer` candidates
  `(src, dst, score)` per step. Proposers: `per_unit` (default, every live
  unit), `per_connection` (every live connection), `global` (one). Neither
  endpoint need be the proposer.
- **Score**: `score(u, src, dst, g)` over candidate pairs from `exhaustive`
  (every windowed pair), `shortlist_per_level`, or `shortlist` (both rank
  units by a user `importance`, M = `shortlist_size`). A boolean predicate
  adapts via `predicate_add_conn` (True -> 0.0, False -> -inf).

Shortlists. `shortlist_per_level` is the canonical shortlist: for each source
level, ascending, the sources are the top-M live units of that level by
importance and the destinations the top-M live units inside that level's
validity window (the `max_level_gap` window intersected with `direction`), by
importance; each level's M x M grid is scored. Every source level gets its own
grid, so no level can take the whole budget and starve the others; it is
levels-based and so topological only. Plain `shortlist` is the degenerate
single global grid (the top-M units of the whole network, crossed), kept for
pipeline mode. Importance ties break by ascending unit id; fewer than M
eligible units give a smaller grid.

Pipeline, in order: trigger (`every_step` default, `every(n)`,
`on_units_added`, `when(g)`); candidates from live proposers; validity (each
failure scores -inf): live in-range endpoints, `src != dst` unless
`allow_self_loops`, `|level(dst) - level(src)| <= max_level_gap` (a growth-rule
attribute — the network-level `neighbourhood` is removed), `direction`
(`any`/`deeper`/`same_or_deeper`); non-finite scores veto; `dedupe_live`
(default **False**) vetoes candidates equal to a live edge; `dedupe_step`
(default **False**) keeps the first of equal keys; per-source-level selection
(`top_k` of `max_new_per_level` / `threshold(g)` / `all`, then
`max_new_per_step` across levels) in the total order; claim of free slots in
order (drops raise `overflow`); `init` with declared field defaults; flags
(`needs_resort`, `grown`). Without dedupe, duplicate candidates create
parallel edges — by design; live dedupe costs a sort of live keys per step.

Total order: sort key `(-score, src, dst, candidate_index)` ascending.
Candidate index: per_unit `unit_id * P + j`; per_connection `rank * P + j`
(rank = ascending `(src, dst, occurrence)` among live connections); global
`j`; exhaustive `src * capacity + dst`; shortlist row-major over the
importance-ranked grid (importance ties break by ascending unit id);
shortlist_per_level `level_rank * M * M` plus the row-major position in that
level's grid (level_rank counts the levels holding live units, ascending).

Randomness: proposal rules receive an `Rng` keyed by
`(Network.seed, state.step, stream=1, proposer_key, j)` per the normative
contract in `plastax.rng`'s module docstring, pinned bit-exactly by the
`rng_philox32.json` golden shared with plastax-cpp.

The golden files under `tests/golden/` tagged `growth_v2` encode this
pipeline's expected outputs case by case and flip from skipped to enforced as
the stages land.
