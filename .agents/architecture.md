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
| `Loss[GS]` | `per_output(u,i,target,g)→(scalar, UnitWrite)` | output units |
| `UpdateConn[GS]` | `incoming(...)→ConnWrite`, `outgoing(...)→ConnWrite` | the edge (two-pass) |
| `PruneConn[GS]` | `predicate(u,c,cid,g)→Bool` | tombstones edges |
| `AddConn[GS]` | attr `max_candidates:int`; `score(u,src,dst,g)→Float`, `init(u,src,dst,g)→ConnWrite` | grows edges |
| `ResetGlobal[GS]` | `reset(g)→GS` | globals, between episodes |

`AddConn` may *structurally* (via `getattr`, not in the Protocol) also declare
`max_candidate_units:int` + `importance(u,i,g)→Float` to switch from the
`O(num_units²)` full grid to an `O(num_units + M²)` shortlist, and
`shortlist_per_level:bool` for a per-bucket grid.

### Assembly (`phases.py`)

`build_phases(net, static, *, overflow_sink)` (`phases.py:64`) emits an ordered
tuple of pure `state → (state, loss_contribution)` phase functions in the
**fixed order**:

```
forward → loss → backward → update_conn → prune_conn → add_conn → reset_global
```

Each phase is appended **iff its trait slot is not `None`** (forward is
unconditional). This is **phase elision** (invariant #1): an absent phase means
no equations in the jaxpr, verified by `test_phases_elision.py`. Forward and
backward branch on `net.propagation` (single flat sweep for PIPELINE; a
per-bucket level walk for TOPOLOGICAL). `build_add_conn_phase` (`phases.py:426`)
is the most complex: candidate grid (full or shortlisted) → level-window filter
+ dedup vs live edges → `score` → per-bucket `top_k` → prefix-sum free-slot
claim → commit only finite-scored candidates → set `needs_resort` if a
committed edge isn't level-preserving. A `-inf` score is a **hard veto**.

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

### Host loop (`driver.py`)

`Driver.step(inputs)` (`driver.py:51`) runs the jitted step and reacts to the
flags it returns — the **retrace protocol**:

- **overflow** → for each full bucket, `state.grow_bucket` (new `NetworkStatic`
  → `make_step` retrace), retry the *same* inputs against the failed attempt's
  **output** state (the donated input buffers may be gone).
- **needs_resort** → `topo.resort` (new bucket layout, new `NetworkStatic`),
  rebuild the step, return.
- else commit.

`topo.resort` (`topo.py:149`) recomputes levels (`recompute_levels`,
Bellman-Ford relaxation bounded by `kahn_max_depth`), redistributes edges into
new per-level buckets (prefix-sum compacting scatter + stable sort on
`dead*num_units + to_id` to restore the `(dead, to_id)` order the segment
reductions need), and sizes new capacities via `capacity_policy`. It returns a
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
  `FieldSpec.default` **is** the regrow-init: on `AddConn` growth the framework
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

Full detail lives in `tests/README.md`; the contract:

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
