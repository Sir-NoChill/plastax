# plastax sparse plan: native-sparse dynamic sparse training

Coding-agent handoff, structured like `DISTRIBUTION_PLAN.md`: phases with
acceptance criteria, HUMAN markers, and a Deviations section. Builds on Track
A.1 (`plastax.optim`, done: five optimizers validated vs optax, per-connection
state as SoA columns). Commits follow the agent-commit protocol.

## Thesis (locked with the user, 2026-08-21)

Standard dynamic sparse training (DST) -- RigL, SET -- is implemented
**mask-based**: a dense weight matrix plus a binary mask, dense matmuls
throughout, pruned weights merely zeroed by the mask. So its weights,
activations, gradients, **and optimizer state are all O(N^2)**, sparsity or not.

plastax runs the *same DST algorithms* on the SoA edge arena, everything at
**O(E)** live edges. The claim, in order:

1. **Accuracy-per-parameter parity** with mask-based SET/RigL (same connectivity
   dynamics -> same accuracy at the same edge count).
2. **Memory win** over mask-based DST: `O(E)` vs `O(N^2)` for weights + optimizer
   state + activations/grads. This is a claim *against mask-based DST* (the
   standard), not against general sparse-matrix libraries.
3. **Compute win** at high sparsity: `O(E)` work vs `O(N^2)` dense matmul.

Not in scope here: **continual/non-stationary adaptation** is a corollary of the
grow/prune design (the machinery gives it for free) but has no dedicated
algorithm yet -- a demonstration is future work. The detailed **compute-crossover
study** (the exact N x sparsity where wall-clock flips) is deferred; FLOPs/sample
is the primary compute metric until then.

Reality check to respect: `examples/parallel_mnist/benchmark_vs_jax.md` found
plastax ~3x slower online and the dynamic win absent at MODEST sparsity/scale --
a GPU eats the mask/padding cheaply. The win regime is EXTREME sparsity (90-99%+)
at enough scale that `O(E) << O(N^2)`.

---

## Phase S0 -- A.2: sparse optimizer-state init (foundation)

The `add_conn` phase already resets a grown edge's untouched fields to their
`FieldSpec.default` (phases.py, "reset to the FieldSpec default rather than
inheriting whatever a previous occupant left"). So on a regrown edge adam's
`opt/m`/`opt/v`/`opt/t` start at 0 -- exactly RigL's "zero the moments for
regrown weights", and `opt/t = 0` restarts that edge's bias-correction schedule.
No new init code is needed; this phase locks and formalizes the behaviour.

- **S0.1** Test: an adam net with grow+prune policies; prune an edge, grow a new
  one into the freed slot, assert the new edge has `opt/m = opt/v = opt/t = 0`
  and no stale state leaked from the pruned occupant.
- **S0.2** Contract: state in the `Optimizer` protocol docstring and
  `docs/optimizers.md` that `state_fields`' `FieldSpec.default` *is* the
  regrow-init, and growth policies write `WEIGHT` (+ their own fields) only,
  never optimizer columns.

Acceptance: S0.1 passes; the contract is documented. Expected to be
test-and-docs only -- the mechanism already exists.

## Phase S1 -- SET as plastax policies (the algorithm)

SET (Mocanu 2018): magnitude prune + **random** grow. Random growth needs no
gradient over absent edges, so it fits the SoA arena with zero friction (unlike
RigL, below) -- the primary algorithm. Scoped minimally here, ahead of the full
Track B `plastax.heuristics`.

- **S1.1** `PruneConn`: prune the smallest-magnitude fraction (SET's zeta;
  smallest-positive + largest-negative per the paper), reusing the existing
  prune machinery.
- **S1.2** `AddConn`: random regrowth of the count just freed, over the existing
  candidate-shortlist mechanism; new weight ~ small init, optimizer state
  auto-zeroes (S0).
- **S1.3** A `Rewiring` bundle: pair the prune+grow at a cadence (every N steps)
  with a fraction schedule (constant, or cosine decay per RigL).
- **S1.4** `examples/set_mnist.py`: an MLP trained online with adam + SET at a
  target sparsity, holding the live-edge count ~constant.

Acceptance: SET holds a fixed sparsity across training; accuracy climbs;
the run exercises S0 (moments zero on every regrow).

## Phase S2 -- the mask-based reference (the baseline to beat)

A faithful mask-based SET in jax+optax: dense weight matrices + a binary mask,
magnitude prune + random regrow by flipping mask bits, dense matmuls throughout,
adam over the full dense parameters. This is how RigL/SET are standardly written
and the honest comparison target.

- **S2.1** Implement it at matched architecture, sparsity schedule, init, and
  data order.
- **S2.2** Confirm it reproduces the SET/RigL-reported accuracy-per-parameter on
  the task (else the baseline, not plastax, is the confound).

Acceptance: the mask-based baseline reaches published accuracy-per-param.

## Phase S3 -- the comparison (accuracy-per-param, memory, compute)

Same algorithm, same sparsity, plastax-native vs mask-based.

- **S3.1 Accuracy-per-parameter** (lead result): plastax matches mask-based
  across 90-99% sparsity; both track a dense baseline at far fewer params.
- **S3.2 Memory**: count bytes for weights + optimizer state + activations/grads.
  plastax `O(E)`; mask-based `O(N^2)` + mask. Report the ratio vs sparsity -- the
  concrete win *over mask-based DST*.
- **S3.3 Compute**: FLOPs/sample (`O(E)` vs `O(N^2)`) is the primary,
  batching-agnostic metric. Wall-clock is secondary and CAVEATED: plastax is
  online (batch-1) and host-dispatch-bound, so a wall-clock win needs high
  sparsity + scale to amortize (see Risks).

Acceptance: accuracy-per-param parity; a memory ratio that grows with sparsity;
a FLOPs/sample ratio favouring plastax at high sparsity.

## Phase S4 -- analysis + corollaries (deferred detail)

- **S4.1** Compute-crossover study: sweep N x sparsity to locate where plastax's
  `O(E)` wall-clock actually beats mask-based `O(N^2)`, accounting for batch-1
  host overhead. (Later.)
- **S4.2** Continual corollary (no code): note the grow/prune machinery gives
  online capacity re-allocation for free; a non-stationary demonstration is
  future work.

## Risks

- **Batch-1 online** is plastax's speed handicap vs batched mask-based training.
  The clean compute metric is FLOPs/sample; wall-clock parity is a
  high-sparsity/scale claim, not a given. Batching plastax is a separate,
  out-of-scope capability.
- **Modest-scale null result is expected** (prior benchmark): the win only shows
  at extreme sparsity + scale. Choose the operating point deliberately.
- **RigL's gradient growth** needs dense gradients over absent edges, which
  fights the arena -> shortlist-restricted only, and after SET. SET (random
  growth) is the primary algorithm precisely because it sidesteps this.
- **Accuracy-per-param parity is a prerequisite**: if plastax's online SET does
  not match mask-based accuracy at the same sparsity, the memory/compute wins are
  moot. S3.1 gates S3.2/S3.3.

## Sequencing

S0 (lock A.2) -> S1 (plastax SET) -> S2 (mask baseline) -> S3 (compare:
accuracy-per-param, then memory, then FLOPs) -> S4 (crossover + corollary, later).

## Deviations

(Date + rationale, as elsewhere.)
