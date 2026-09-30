# plastax scale plan: in-place churn at O(churn), and a layout backend

2026-09-30 · Status: **in progress** on branch `scale` (worktree `../plastax-scale`). This plan ports the
plastix C++ improvements (in-place growth, radix resort, the correctness fixes)
to plastax, and assesses CSR and dense layouts in JAX as a multi-backend path.
All numbers were measured on cdol01 (RTX 5000 Ada, 32 GB) with jax 0.11.0 +
`jax[cuda13]`, plastax `a145691`, in a scratch venv. The probes are
`docs/scale_plan/plastax_probe.py` (synthetic churn net),
`jax_layouts_probe.py` (one layer per layout) and `rb2.py` (sort strategies);
they become committed benchmark scripts in P0.

## TL;DR

- **plastax already has the in-place design.** Tombstoned slots, per-level
  fixed-capacity buckets, and new edges written into free slots of their own
  level are the same mechanism the C++ `InPlaceGrowth` path added. What it
  lacks is *cost proportional to churn*: the growth phase does O(capacity)
  work, including a **full sort of every bucket, every step**.
- **Measured** at E = 300M, 64 edges churned per level per step:
  - plastax takes **144 ms/step**: forward 29.7, prune 9.8, growth 105.
  - C++ in-place takes 12.9 ms; tuned CSR with a rebuild takes 65.5 ms.
  - Growth is 73% of the plastax step, against 0.01 ms in C++.
- **Memory is plastax's strength.** 300M live edges take 7.0 GB of state,
  against 27 GB for C++ Plastix. That is about 23 B per live edge including
  power-of-two padding (13 B per slot), against 97 B per slot. plastax could
  plausibly reach about 1B edges on this card, where C++ stops at about 300M.
- **Feasible, without breaking an invariant** for the churn work (P0-P3). The
  projected step is about 40 ms at 300M after P1 and about 24 ms after P2.
  That would be 2.7× faster than a CSR rebuild and 1.9× slower than C++ in
  place.
- **CSR in JAX works, but only with a flag, and only pays off batched.**
  - `jax.experimental.sparse.BCSR` lowers to cuSPARSE only with
    `jax_bcoo_cusparse_lowering=True`. Without it, it is 30-80× slower than a
    plain `segment_sum`.
  - With it, CSR is 1.2× faster at batch B = 1, and **9.5× faster at B = 128**.
  - A CSR rebuild (radix sort) costs 19 ms at 25M edges, so CSR must be a
    cached, derived view, not the arena.
- **Dense only pays at ≥ ~10% density, and only fits small layers.** A 16K ×
  16K layer is 1.1 GB. A 158K × 158K layer would be 100 GB.
- **Three correctness issues** were found on the way (P0). One is a real
  bug: the growth phase's duplicate check overflows int32 once
  `num_units > 46,340`.

![One step at 300M edges](docs/scale_plan/step_300M.png)

## 1. What the C++ work changed, and where plastax stands

| C++ change (plastix fork) | plastax today | Port? |
|---|---|---|
| In-place growth into same-level dead slots (`5e0aff2`) | Same mechanism (`build_add_conn_phase` prefix-sum slot claim) | Already there. Make it O(churn): **P1** |
| Headroom dead slots per level; resort only on overflow | `capacity_policy(headroom=)` exists but is rounded to a power of two. The driver grows the bucket and retraces on overflow | Fractional, non-power-of-two capacities: **P2** |
| Fused radix resort with a 32-bit key and level-bits cap (`281f05c`, `8f57838`, `3f1eb15`) | `topo.resort` does one cumsum-scatter plus one u32 `sort_key_val` per bucket, which is already CUB radix | Minor; resort is rare once growth is in place. Fold buckets into one keyed sort only if profiling says so |
| Key-width overflow fix (`3f1eb15`) | **Same bug class:** `from*num_units+to` in int32 in the add_conn duplicate check | **P0 bug fix** |
| Sampling salt: proposals must depend on the step (`3c24c83`) | The hash scores take `g["step"]` or a cursor; the policy is responsible | Audit, document, and test in P1's proposal API |
| Stale CUDA-graph fix (`RangesEpoch_`) | N/A: `NetworkStatic` is the jit key, and state is functional | none |
| Batched device forward (`3df28ae`) | One sample per step. COO `segment_sum` at B = 128 is 9.5× slower than CSR | Layout backend: **P5** |
| `--validate-every` host-replay check | Oracle tests exist; no "in place vs rebuilt" equivalence test | **P1 test gate** |

### Measured phase costs (ms/step, fwd = forward only; + adds one phase)

| E (live) | L per layer | slots | state | fwd | + prune | + growth | full step |
|---|---|---|---|---|---|---|---|
| 5.4M | 16,384 | 8.4M | 0.11 GB | 0.45 | +0.10 | +1.34 | 1.88 |
| 50M | 158,114 | 67M | 0.88 GB | 4.16 | +1.14 | +9.5 | 15.1 |
| 300M | 387,298 | 537M | 7.0 GB | 29.7 | +9.8 | +108 | 144 |

The C++ in-place step at the same sizes: 0.24, 2.19 and 12.9 ms. A shortlist
of M = 1024 instead of 64 at 50M changed the growth phase by less than 1 ms, so
the growth cost is the O(capacity) work, not the candidate grid.

### Where the growth time goes (`phases.py:680-888`, per bucket, per step)

1. **Duplicate check:** `jnp.sort(live_pair)` over the whole bucket capacity,
   then `searchsorted`. This sort *is* the 105 ms: it is the per-step rebuild the
   C++ work removed, reintroduced as a duplicate filter.
2. **Free-slot rank:** `cumsum(dead)` plus a capacity-sized scatter building
   `local_slot_for_rank`. This is O(capacity) and bandwidth-cheap, but still an
   extra pass.
3. **Candidate grid:** M × M per level (with a shortlist), or `num_units²`
   (without one). The plastax DeepR port (`15_deepr_multimnist/plastax/deepr.py`)
   has no shortlist, so it scores the full `num_units²` grid. At the C++ DeepR's
   4.4M hidden units that is about 2·10¹³ candidates, so today it cannot run
   beyond a few thousand units.

## 2. Correctness findings (P0)

1. **int32 pair-id overflow in the growth phase's duplicate check**
   (`phases.py`, `live_pair = from * num_units + to`, int32). Once
   `num_units > 46,340` the ids wrap. Distinct pairs then collide (a false
   "duplicate" drops a valid candidate), or a real duplicate is missed and a
   parallel edge is grown. Every probe here was in this regime (the smallest
   has 49,152 units). Fix: compare `(src, dst)` as a pair (two-key search) or pack the key
   into a per-bucket local id space. Add a regression test with `num_units`
   above 2¹⁶.
2. **The `indices_are_sorted=True` hint on the topological forward is
   violated.**
   - The hint is set in `phases.py:_build_forward_topological_phase`, but it
     only holds right after construction or a resort.
   - Prune tombstones in place, which redirects dead targets to `num_units` in
     the middle of the array. Growth writes into arbitrary dead slots.
   - Measured: about 1,300-1,500 descents per bucket after 20 churn steps. The
     output still matched an unsorted `segment_sum` to 2e-6 on GPU (XLA's
     scatter-add ignores the hint there).
   - It is still undefined by contract, and could silently break on CPU or TPU,
     or in a future XLA.
   - The hint also buys nothing: at 25M edges the unsorted `segment_sum` was
     *faster* (1.49 vs 1.73 ms).
   - Fix: pass `False`. If a sorted fast path is ever wanted, gate it on a
     static "freshly resorted" layout, never on a runtime property.
3. **Host sync every step.** `Driver.step` calls `bool(result.overflow)` and
   `bool(state.needs_resort)` on every step, which blocks dispatch.
   - Overflow drops candidates rather than corrupting state, and
     `needs_resort` is already sticky in state.
   - So both can be made sticky and checked every N steps, with N = 1 kept as
     the exact current behaviour.

## 3. Decisions (2026-09-30)

1. **CSR only, no dense view.** Invariant 8's densification exclusion stays.
   A custom **Pallas** kernel is in scope now, as the extension avenue beyond
   cuSPARSE.
2. **Duplicate edges are allowed.**
   - Proposal growth defaults to no duplicate check.
   - An opt-in exact per-step dedupe exists, and algorithm developers are told
     plainly (docstrings and user docs) that duplicates are possible.
   - Performant algorithms exclude duplicates by construction.
3. **Batched inputs are in v1**, documented as a convenience: the library stays
   primarily streaming, one sample per step.
4. **A `cuda13` extra** is added next to `cuda12`.

## 4. Plan

Every task ends with the fast suite green, `mypy --strict`, and the hooks
passing. A GPU number goes in the loop log when the task touches performance.
One scope per commit (SCOPES.md).

### P0: correctness and harness

- [ ] 0.1 `build(packaging)`: add a `cuda13` extra and a `gpu13` alias, plus a
  TOOLING note.
- [ ] 0.2 `fix(phases)`: fix the int32 pair-id overflow in the growth phase's
  duplicate check. Add a regression test with `num_units` above 2¹⁶ where
  wrapped ids collide.
- [ ] 0.3 `fix(phases)`: drop the `indices_are_sorted=True` hint on the
  topological forward and backward. Add a test that churns and then compares
  the forward against an unsorted reference.
- [ ] 0.4 `feat(examples)`: add `examples/benchmarks/` with the churn probe and
  the layout probe (GPU-only, not collected by pytest), plus a README on how to
  run them.

### P1: growth at O(churn)

- [ ] 1.1 `feat(traits)`: add an optional `AddConn.propose(u, j, g) ->
  (src, dst, ok)` for proposal `j` of `max_candidates`.
  - It is O(k) sampling with no candidate grid.
  - Seed with step-dependent state (the C++ salt lesson).
  - Validation: the `score` path and the `propose` path are mutually
    exclusive.
- [ ] 1.2 `feat(traits)`: add `AddConn.dedupe ∈ {"exact", "none"}`.
  - The grid path keeps `"exact"` (today's behaviour). The propose path
    defaults to `"none"`.
  - `"exact"` on the propose path checks proposals against the live pairs of
    their bucket, which keeps the O(capacity log capacity) sort.
  - Docstrings state the multigraph semantics.
- [ ] 1.3 `feat(phases)`: build the propose path.
  - Proposals are routed to the bucket of their source level, with a level
    check that vetoes a proposal breaking the leveling invariant, or sets
    `needs_resort` as today.
  - The slot claim uses `searchsorted` on `cumsum(dead)` for the first k free
    slots, with no capacity-sized scatter. It must stay Scheme-A-aware.
- [ ] 1.4 `test(phases)`: equivalence test in the spirit of the C++
  `--validate-every`.
  - Run N in-place churn steps, then rebuild with `from_edges` from the live
    multiset.
  - Assert the same live multiset and the same forward output. Cover
    overflow → grow and resort.
- [ ] 1.5 Bench, outside this repo: switch the DeepR plastax port to
  `propose`, run it on the real MultiMNIST stream, and scale to the memory
  bound.

### P2: capacity and forward bandwidth

- [ ] 2.1 `feat(topo)`: `capacity_policy(..., align=)` with no power-of-two
  rounding (alignment keeps Scheme-A divisibility). Thread it through
  `from_edges` / `grow_bucket` / `resort`.
- [ ] 2.2 Profile the edge-list forward with nsys: materialisation of the
  vmapped `map` output, and gather/scatter fusion. Fix what is fixable in XLA.
- [ ] 2.3 Find the largest E that fits on 32 GB, and record it.

### P3: driver without per-step sync

- [ ] 3.1 `feat(driver)`: add `Driver(check_every=N)`.
  - Overflow becomes sticky in state (a design check first: adding a
    `NetworkState` field is a pytree change).
  - Check every N steps; the default N = 1 is unchanged behaviour.

### P4: batched inputs

- [ ] 4.1 Design note first.
  - Batched `StepInputs` of shape `(B, num_inputs)`, and unit columns of shape
    `(B, num_units)` during forward and backward only.
  - Connection updates reduce over B. Pinning this down (is it the mean of
    per-sample deltas, or something else?) is the hard part, since optimizer
    state must see one update per step.
  - Decide the API: `make_step(..., batch=B)` or a separate
    `make_batched_step`.
- [ ] 4.2 Build it: a vmapped forward/backward over the unit axis, with shared
  connections.
- [ ] 4.3 Tests: B = 1 matches the streaming step; B > 1 matches the mean of B
  streaming gradient computations for SGD.

### P5: CSR layout view (cuSPARSE)

- [ ] 5.1 Design note: a static per-bucket `BucketLayout ∈ {EDGE_LIST, CSR}` in
  `NetworkStatic`.
  - Only for a built-in `LinearForward`/`LinearBackward`.
  - The view is `perm/indices/indptr` plus the values `weight[perm]`
    (tombstones give 0), plus a COO delta for edges grown since the last
    rebuild, plus a rebuild every R steps or on delta overflow.
- [ ] 5.2 Build the forward (cuSPARSE `csrmv`/`csrmm` via
  `jax.experimental.sparse`, with the lowering flag scoped to the call).
- [ ] 5.3 Backward: the transpose via cuSPARSE, or a CSC view.
- [ ] 5.4 Tests against the edge-list path; benchmark at B = 1 and B = 128.

### P6: Pallas kernel (extension avenue)

- [ ] 6.1 Spike: a Pallas GPU (Triton) kernel for the edge-list forward.
  - The user's per-edge `map` is traced inside the kernel, combined with a
    named monoid via atomics, over one bucket.
  - Measure it against XLA `segment_sum`.
- [ ] 6.2 If it wins: an edge-once batched variant (each edge read once, all B
  samples).
- [ ] 6.3 Place it as an opt-in backend next to CSR.

### P7: parity and write-up

- [ ] 7.1 A plastax implementation in `plastix-synth-bench`, rerun at 5.4M,
  50M and 300M, with plastax rows added to the figures.
- [ ] 7.2 Results Markdown with PNG figures. Keep this plan's log current.

## 5. Autonomous loop protocol

Each iteration runs four discrete steps, in order, and appends one entry to
the log below:

1. **Build:** take the next unchecked task, implement it, add tests, run the
   hooks' checks, and commit.
2. **Document:** update docstrings, docs and plan text touched by the build;
   remove stale comments; commit separately (`docs(scope)`).
3. **Explore:** run a GPU experiment tied to the phase (a measurement, a spike,
   or an idea test). Record the number and what it implies.
4. **Plan:** tick the boxes, re-order or add tasks from what was learned, and
   put anything the user should weigh under "Surfaced for review".

## 6. Surfaced for review

- **Memory headroom beyond C++.** plastax holds 300M edges in 7.0 GB, against
  27 GB for C++ Plastix. The C++ side's 97 B per slot is mostly scratch
  (radix buffers, keys, perm) kept resident. It could probably be cut to about
  30 B, which would let C++ reach about 1B edges too.
- **XLA autotuner noise.** At large E, XLA's sort autotuning tries allocations
  of 9 GB to 4.9 TiB, fails, and falls back. The runs are correct, but the
  logs are noisy and the first compile is slow. It is worth documenting an
  `XLA_FLAGS` setting.
- **`jax.experimental.sparse` needs `jax_bcoo_cusparse_lowering`** to use
  cuSPARSE at all. Any user-level BCSR comparison without the flag is
  30-80× pessimistic.

## 7. Loop log

(Entries are appended per iteration, newest last.)

## Deviations

(none yet)
