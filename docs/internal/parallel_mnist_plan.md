# Parallel MNIST in the plastax paradigm — implementation plan

Audience: a coding agent (and us) implementing the parallel-MNIST structure
study on top of plastax. Reproduces the dense vs. block-sparse baselines from
`edan-phd-research/phd/structure_search` and then goes past them: the oracle
block-diagonal structure is *learned*, not hand-wired, using plastax's
`PruneConn` + `AddConn` tools.

Read alongside: `examples/mlp_xor.py` (the trained-MLP trait template),
`src/plastax/traits.py` (trait protocols), `src/plastax/topology.py` (topology
combinators), and `src/plastax/driver.py` (retrace protocol).

---

## 1. The task (faithful restatement)

`ParallelMNISTStream` (phd `structure_search/data.py`), with the config values
`n_tasks = K = 5`, `n_layers = 3`, `use_bias = false`, online `batch_size = 1`:

- **Input**: K independently-drawn MNIST digits, each flattened to 784, concatenated → **3920** values per step.
- **Output**: K × 10 = **50** logits (a 10-way head per sub-task).
- **Loss**: per-task softmax cross-entropy, `-mean_k log softmax(logits_k)[label_k]`.
- **Accuracy**: per-task argmax vs. label, averaged over the K tasks.
- **Stationary** (`permute_period = 0`): each task has a fixed (identity) label permutation.
- **Non-stationary** (`permute_period ∈ {2000…50000}`): every `permute_period` steps, **one** randomly chosen task has its label→class map replaced by a fresh permutation. The other K−1 tasks are undisturbed — this locality is what makes block structure pay off.
- **Optimizer**: Adam, online (one example per step). Metrics: `average_loss`, `asymptotic_loss/accuracy` (mean over the final 10% of logged steps).

### The one identity that makes this a natural plastax problem

> **A plastax connection *is* a parameter.** `use_bias=false` ⇒ every trainable
> weight is exactly one edge. So "equal parameter budget" (the notebook's x-axis)
> is literally **equal live-edge count** (`live_conn_count`), and "structure
> search at fixed budget" is **prune + grow at fixed arena capacity**.

| | Dense MLP | Block-sparse oracle |
|---|---|---|
| Layer weights | `(H,3920)`, `(H,H)`, `(50,H)` | K× `(h,784)`, `(h,h)`, `(10,h)` |
| Edges (= params) | `3920H + H² + 50H` | `K(784h + h² + 10h)` |
| Cross-task edges | all present, must learn to zero | none, by construction |
| Per-task width at equal budget | small `H` | large `h` (budget ÷ K, fan-in 784) |

At `target_params = 2¹⁹`: dense `H ≈ 130`; oracle `h ≈ 128` per block (`K·h ≈ 640`
hidden/layer). The oracle spends its budget on width instead of cross-task fan-in.

---

## 2. Three networks

1. **`DenseMLP`** — fully-connected DAG. Reproduces the dense baseline.
2. **`BlockSparseOracle`** — hand-wired block-diagonal (K independent sub-MLPs). Reproduces the colleague's oracle.
3. **`DynamicSparse`** — starts over-connected (or random-sparse) at a fixed edge budget; `PruneConn` + `AddConn` discover the structure online. **This is plastax's contribution.**

All three share one trait harness (§3) and one data/metrics harness (§4); they
differ only in initial connectivity (§5) and whether the plasticity phases are
attached (§6).

---

## 3. Shared trait harness

Modeled directly on `examples/mlp_xor.py`, generalized from XOR/sigmoid/MSE to
MNIST/ReLU/softmax-CE. Extra unit fields: `GRAD_PRE_ACT`, `LOSS_GRAD` (as in
mlp_xor) plus one **`IS_LINEAR`** bool marking output (logit) units.

- **`ReLUForward` (ForwardPass)**: `map = weight * activation[src]`, `combine = sum_`,
  `apply`: `where(u[IS_LINEAR,i], acc, relu(acc))` — hidden units ReLU, output units linear logits.
- **`ReLUBackward` (BackwardPass)**: `map = weight * grad_pre_act[dst]`, `combine = sum_`,
  `apply`: `grad_pre_act = (acc + u[LOSS_GRAD,i]) * where(IS_LINEAR, 1, (a>0))`.
  (Output: linear derivative 1; hidden: ReLU derivative `a>0`. Same `LOSS_GRAD`
  bridge mlp_xor uses to inject the loss-layer gradient at the output level.)
- **`SoftmaxCELoss` (Loss)** — the key mapping. `per_output` receives the full
  `UnitView`, so output unit `i` in task `k` reads its 9 sibling logits (their
  ids are static, fixed at build) to form the per-task normalizer:
  ```
  lse_k   = logsumexp(a_j for j in task_k_output_ids)      # read via UnitView
  softmax_i = exp(a_i - lse_k)
  loss_i  = -target_i * (a_i - lse_k)     # target_i = one-hot indicator
  return loss_i, UnitWrite.of((LOSS_GRAD, softmax_i - target_i))
  ```
  Summed over a task's 10 outputs this is exactly `CE_k` with gradient
  `∂L/∂a_i = softmax_i − onehot_i`. **No new trait or framework change needed** —
  the per-output decomposition is exact because the loss phase loops output ids
  in Python with the whole unit view in scope. `StepInputs.targets` carries the
  50-length one-hot (per task), assembled from the K labels host-side.
- **`SgdUpdateConn` (UpdateConn)**: `w -= lr * grad_pre_act[dst] * activation[src]`
  in the incoming pass; outgoing no-op (identical to mlp_xor).
- **Adam extension** (to match the colleague exactly): add per-connection
  `M`, `V` fields (`extra_conn_fields`) and a global step counter in `GS`;
  `incoming` does the bias-corrected Adam update; a `ResetGlobal`/counter phase
  advances `t`. **Recommendation:** land SGD first (M1–M3), add Adam once the
  learning curves are qualitatively right.

Propagation: **topological** for the two static baselines (levels = layers,
full forward per step — the correct, low-latency choice for a fixed DAG).

---

## 4. Data + metrics harness (framework-agnostic, host-side)

- Port a minimal `ParallelMNISTStream`: load MNIST once, per step sample K
  indices, concat 784-blocks → 3920 input vector, apply per-task label
  permutations, emit `(inputs, one_hot_targets, labels)`.
- Non-stationary: every `permute_period` steps, re-permute one random task's
  label map (identical to `_advance_permutations`).
- Drive with `Driver` (owns overflow-grow + resort retrace). Log `loss`
  (`result.loss / K`), running/asymptotic loss, and argmax accuracy per task.
- Start at the **50k-param small baseline** (`H≈13` dense, `h≈12`/block) for a
  fast, laptop-CPU-tractable first repro; scale to `2¹⁹` on GPU / Scheme-A
  sharding (§7).

---

## 5. Topology construction

- **Dense**: `topology.sequential(input_units(3920), dense(3920,H), dense(H,H), dense(H,50))` — already supported.
- **Block-sparse oracle**: needs a **new combinator** in `topology.py`,
  `parallel_blocks(*branches)` (or `block_diagonal`), that lays K independent
  sub-topologies into disjoint unit-id ranges with **no cross edges**, sharing
  the concatenated input/output id spaces. Each branch is
  `sequential(input_units(784), dense(784,h), dense(h,h), dense(h,10))`. This is
  a small, well-scoped addition mirroring `sequential`'s offset bookkeeping and
  is independently useful. Output ids are grouped by task for `SoftmaxCELoss`.
- **DynamicSparse**: build the oracle-or-dense skeleton, then either start from
  a random sparse subset at the target edge budget, or start dense and let
  pruning carve it down. Arena capacity = the fixed parameter budget.

---

## 6. The contribution: *learning* the block structure

Attach plasticity to a `DynamicSparse` net (fixed edge budget = fixed params):

- **`PruneConn`**: magnitude pruning — tombstone edges with `|weight| < θ`
  (θ fixed, annealed, or a percentile). Cross-task edges, being useless for
  parallel MNIST, decay under the gradient and get pruned first.
- **`AddConn`**: `max_candidates`-bounded growth over scored candidate edges.
  Score options (a genuine design axis to sweep): weight-gradient magnitude
  (`|∂L/∂w| = |grad_pre_act[dst]·activation[src]|`, a Rigging-the-Lottery-style
  criterion), a Hebbian/correlation score, or utility. Growth refills freed
  budget, concentrating capacity where it reduces loss — i.e. **within-block**.
- **Invariant to track**: fraction of live edges that are cross-task, over
  training. The thesis is it **→ 0**, recovering block-diagonal connectivity
  from no structural prior.

### Pipeline vs. topological for the dynamic net (decision point)

`AddConn` that grows a level-violating edge sets `needs_resort` → the `Driver`
resorts and **retraces**. A constantly-rewiring net can retrace often. The
driver docstring's own guidance applies: *frequent retrace ⇒ prefer a pipelined
formulation* (single flat bucket, arbitrary execution order, **no resort ever**).

- **Recommendation**: static baselines → **topological**; `DynamicSparse` →
  **pipeline**, accepting one-hop-per-step latency (a 3-layer signal takes 3
  steps to traverse — fine for a long online stream, or run a few pipeline
  sweeps per input). Keep a topological+resort variant as a comparison to
  *measure* the retrace cost the pipeline mode avoids. This is exactly the
  trade-off the freshly-expanded driver docstring describes.

### Why this can *beat* the oracle

1. **No prior needed** — matches oracle capacity efficiency without being told the block structure.
2. **Non-stationary adaptation** — when one task's labels permute, prune+grow can re-route that block's capacity online; a fixed oracle cannot restructure. Directly targets the notebook's "faster adaptation" hypothesis, with a stronger claim.
3. **Beneficial cross-task sharing** — if any feature genuinely transfers across tasks, the dynamic net can *keep* that cross edge where the block-diagonal oracle forbids it — a structure strictly richer than block-diagonal.

This is the JAX-native, GPU-parallel realization of the phd repo's own
`DynamicNetwork`/`ConnectivityManager` (`dynamic_stationary.yaml`), now
expressed as first-class plastax traits.

---

## 7. Scale & the multi-GPU backend

These nets are edge-heavy (dense first layer alone ≈ `3920·H` edges; ~500k at
`2¹⁹`). This is precisely what the current Scheme-A work targets: connections
are sharded edge-wise across the mesh while units replicate, and the monoid
collective all-reduces per-shard partial accumulators (`shard.py`, `sweep.py`
collective, `examples/sharded_reservoir.py`). Plan of attack: correctness at 50k
on CPU → `2¹⁹` on a single GPU → Scheme-A shard across GPUs for the full sweep.

---

## 8. Milestones & acceptance criteria

| M | Deliverable | Acceptance |
|---|---|---|
| **M0** | Data stream + metrics + shared traits (ReLU fwd/bwd, SoftmaxCELoss, SGD) | Single-task MNIST (K=1) trains to sane accuracy; CE + grad match a numpy reference |
| **M1** | `DenseMLP`, stationary, 50k budget | Learning curve descends; matches dense baseline qualitatively |
| **M2** | `parallel_blocks` topology + `BlockSparseOracle`, stationary | At equal live-edge budget, oracle asymptotic loss ≤ dense (capacity-efficiency result reproduced) |
| **M3** | Non-stationary stream, both baselines | Oracle keeps lower average loss across `permute_period`; gap widest at short periods (notebook §non-stationary reproduced) |
| **M4** | `DynamicSparse` (prune+grow), stationary | Recovers ≈ block-diagonal (cross-task edge fraction → 0); asymptotic loss ≈ oracle from a dense/random start |
| **M5** | `DynamicSparse`, non-stationary | Matches/**beats** oracle average loss; faster recovery after a permutation event; report retrace/overflow behavior (pipeline vs. topological) |
| **M6** (opt.) | Adam parity + `2¹⁹` scale-up on GPU/Scheme-A | Reproduces the colleague's absolute numbers; scales edge-wise-sharded |

Each milestone is one runnable example under `examples/` (or a small
`benchmarks/parallel_mnist/` package) plus a focused test, following the repo's
per-file review cadence.

---

## 9. Open decisions / risks

- **Adam now or later?** Colleague uses Adam (block-sparse optimum LR ~8× dense).
  SGD-first keeps M1–M3 simple but won't match absolute numbers — recommend SGD
  for structure results, Adam for M6 parity.
- **AddConn score function** — real research axis; start with gradient-magnitude, keep it swappable.
- **Pipeline latency vs. topological retrace** for the dynamic net (§6) — recommend measuring both.
- **Prune threshold schedule** — fixed vs. annealed vs. percentile; affects how fast structure converges and whether budget stays saturated.
- **MNIST data dependency** — need the raw MNIST arrays available offline to the plastax repo/tests (vendor a small loader; keep the full dataset out of git).
- **`parallel_blocks` placement** — extends `topology.py`; small and reusable, but is a genuine public-API addition (docstring gate applies).

---

## 10. First concrete step

Implement **M0** end to end at **K=1, 50k budget** — it exercises every shared
component (ReLU forward/backward, softmax-CE-per-output, SGD update, the data
stream, metrics) on the simplest possible instance, and validates the
softmax-CE-in-`per_output` decomposition against a numpy reference before any of
the multi-task or structural machinery is added.
