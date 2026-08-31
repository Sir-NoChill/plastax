# plastax RL plan: topology modification under non-stationarity

Implementation plan for four rewiring baselines — **CBP**, **GMP**, **NE**,
**RLx2-λ** — run against a streaming deep-RL learner. Sibling to
`SPARSE_PLAN.md`, which this reuses wholesale. Read `AGENTS.md` first; scopes
come from `SCOPES.md`.

---

## Decisions (locked with the user, 2026-08-30)

1. **Streaming RL, not batching.** Adding a batch axis to plastax is an
   explicit **non-goal** for the system in its current state. Every RL result
   here is produced at batch 1.
2. **CBP is the entry point.** It carries the most interesting open question
   (a local formulation of a per-layer ranking) and needs no topology change,
   so it isolates cleanly.
3. **Stage 0 is part of the plan**, not a preamble. Specified below.
4. **RLx2 becomes RLx2-λ.** A successful streaming adaptation is a goal in its
   own right — it would be the representative example of the whole family
   surviving the move to batch 1. Success criterion is defined below, and it is
   not "matches the paper's numbers".
5. **NE runs capped and uncapped.** The growth trajectory is the object of
   interest.
6. **ObGD is built in `examples/`**, promoted to `src/plastax/optim/` only if it
   comes out clean.
7. **CBP v1 (the two-hop local threshold) is in scope.**

---

## Thesis

`SPARSE_PLAN.md` claims mask-based DST keeps weights, gradients and optimizer
state dense at `O(N²)` while plastax runs the same algorithms on a live-edge
arena at `O(E)`. That claim was validated on a **stationary supervised** signal
(CIFAR-10 frozen-conv head, holds to 94 % sparsity).

This plan extends it to the setting the DST-in-RL literature says is the hard
one: **bootstrapped, non-stationary targets**. Two claims to test.

1. **Capability.** Every rewiring rule below is expressible as per-unit-local
   plastax traits with zero global state — including two (CBP's contribution
   utility, NE's dormancy) that the literature only ever writes as dense mask
   ops. If that holds, the O(E) arena is not a compression trick, it is the
   natural representation for this whole family.
2. **Transfer.** Graesser et al. (ICML 2022) found gradient-guided regrowth
   *underperforms* magnitude pruning in DRL — the RigL signal degrades under
   bootstrapping. Our 94 % result was measured on a stationary signal. **We
   should expect it to move.** Where it moves is the finding.

Non-goal, restated from `SPARSE_PLAN.md`: beating dense matmul wall-clock. RL
critics are 256×256 MLPs; per `project_plastax_perf_ceiling` those sit far on
the wrong side of the 99.6 % break-even. The claim is expressiveness of the
rule, not throughput. Report wall-clock in the same table as the capability
result and say why.

---

## The binding constraint: batch-1

`plastax.StepInputs.inputs` is `Float[Array, " num_inputs"]` — **one sample per
step**, no batch axis. The state pytree is donated and updated in place, so the
step cannot be `vmap`ped over a batch without replicating the arena.

Standard TD3/DQN does one gradient update per env step at batch 256. On a 1M-step
MuJoCo run that is 256M sequential plastax steps: at an optimistic 200 µs/step,
~14 h **per seed per method per env**. Not viable, and per decision 1 we are not
fixing it by changing the library.

### Streaming RL

The **stream-x** family — Elsayed, Vasan & Mahmood, *Streaming Deep
Reinforcement Learning Finally Works* (NeurIPS 2024), `arXiv:2410.14606`,
reference implementation at `github.com/mohmdelsayed/streaming-drl`. stream-AC(λ),
stream-Q(λ) and stream-SARSA(λ) learn MuJoCo, DM Control and Atari at
**batch size 1, with no replay buffer and no target networks**, via the ObGD
optimizer, eligibility traces, LayerNorm and sparse init. Each algorithm is
~150 lines.

This is not a workaround. It is a 256× reduction in plastax steps that makes the
study affordable (1M env steps ≈ 1M plastax steps ≈ minutes), and its shape —
online, one sample, sparse init, per-edge traces — is exactly plastax's. It is
also same-lab work as CBP, which matters for how the comparison reads.

---

## Where this sits: the streaming RL landscape

Surveyed 2026-08-30. Two literatures, and **the intersection is empty.**

**Streaming RL, since stream-x.** The line is active and almost entirely
Alberta + Mila + DeepMind:

- *Streaming Deep RL Finally Works* — Elsayed, Vasan & Mahmood, NeurIPS 2024,
  `arXiv:2410.14606`. ObGD, eligibility traces, LayerNorm on pre-activations
  without learnable scale/bias, **SparseInit**.
- *Intentional Updates for Streaming RL* — Sharifnassab, Elsayed, De Asis,
  Mahmood & **Sutton**, `arXiv:2604.19033` (Apr 2026). Specifies an intended
  change in *output* units and solves for the step size, rather than specifying
  a parameter-space step size. Intentional TD(λ) / Q(λ) / PG, all with
  RMSProp-style diagonal scaling and traces. Beats ObGD; notes ObGD permits
  large overshoots when gradients align across timesteps.
- *Squeezing More from the Stream* — Nilaksh, Clavaud, Reymond, Rivest &
  Chandar (Mila), `arXiv:2602.09396` (Feb 2026). SPR auxiliary loss plus
  orthogonal gradient projection. States plainly: *"not many works study
  representation learning in a streaming fashion, and none do for streaming RL,
  making this an open challenge."*
- *Revisiting Adam for Streaming RL* — Gogianu, Lutu & Pascanu,
  `arXiv:2605.06764` (May 2026). Two properties matter: bounded objective
  derivative and variance-adjusted updates. C51 has both; Adaptive Q(λ) across
  55 Atari games.
- *Towards Batch-to-Streaming Deep RL for Continuous Control* — De Monte,
  Cederle & Susto, `arXiv:2603.08588` (2026). S2AC and SDAC; motivated by tiny
  robotics and on-device finetuning, 2×128 nets.

**The load-bearing fact for us: stream-x already ships a sparse network.**

> *"We use a simple technique to introduce sparsity at initialization by
> randomly initializing most weights to zeros. Specifically, we impose a
> sparsity level s (e.g., 0.9) at each layer representing the proportion of
> zero-initialized weights."*

Their own ablation: replacing SparseInit with LeCun init leaves the agent able
to learn "but slower". Every streaming follow-up above inherits it — S2AC/SDAC
adopt it explicitly. **So streaming RL is already running at ~90 % sparsity, and
that sparsity is static and never rewired.** The regime is ours; the mechanism
is missing.

**Dynamic sparse training in RL, meanwhile, universally assumes replay.**
Sokar et al. (IJCAI 2022), Graesser et al. (ICML 2022), RLx2, MAST, NE and MST
all evaluate on batched, replay-buffer agents. A targeted search for DST or RigL
combined with eligibility traces or a no-replay setting returns nothing.

**Therefore the opening is precise:** *nobody has made streaming RL's sparse
network dynamic.* That is a one-sentence contribution statement, it sits exactly
on plastax's O(E) arena, and stream-x's own ablation already establishes that
sparsity matters in this setting — we are not arguing for the premise, only
supplying the mechanism.

**Two cautions from the same literature.**

- Sokar et al. found SET in **online** RL destabilized past 50 % sparsity.
  Streaming is online by definition and stream-x sits at 90 %. Expect trouble in
  exactly the band we care about; the LayerNorm and reward/observation scaling
  that make stream-x stable may be what changes this, and that is a testable
  claim rather than an assumption.
- *A Closer Look at Sparse Training in Deep RL* reports that **disabling bias
  terms substantially improves RigL at high sparsity**, aligning it with
  pruning. Cheap to adopt; make it a config flag from the start. (Sourced from a
  search summary, not the full paper — verify before citing.)

**Name collision to avoid:** `arXiv:2505.01584`, "plasticity preservation for
DRL in adaptive *video* streaming", is about bitrate control, not streaming RL.

---

## What already exists (reuse map)

Nothing below needs a new trait. Read these before writing anything.

| Need | Already in the tree |
| --- | --- |
| per-unit magnitude prune threshold | `MagnitudeStats` / `SetPrune`, `examples/dst_sparse.py` |
| runtime-annealable prune fraction | `AnnealedMagnitudeStats`, `ZETA` column, `set_zeta`, `cosine_zeta`, `examples/xmc.py` |
| gradient-scored growth on absent edges | `GradientGrow` (RigL), `examples/dst_sparse.py` |
| scalable candidate grid | `max_candidate_units` + `shortlist_per_level`, `phases.build_add_conn_phase` |
| per-unit reduction over **incoming** edges | `ForwardPass.map`/`apply` |
| per-unit reduction over **outgoing** edges | `BackwardPass.map`/`apply` (src/dst inverted vs forward) |
| arena with room to grow into | `NetworkBuilder.from_edges(capacity_headroom=…)` |
| per-edge optimizer state, any dtype | `optim.Optimizer.state_fields` |
| separate train / churn / eval nets over one state | `make_net(mode=…)` + three `make_step`s, `examples/dst_sparse.py` |
| stateless per-unit randomness | `_hash01(src, dst, cursor)`, `examples/dst_sparse.py` |
| teacher task + online eval loop | `teacher_task`, `_sample`, `evaluate`, `examples/dst_sparse.py` |

Phase order is fixed: `forward, loss, backward, update_conn, prune_conn,
add_conn, reset_global`. A **churn** net may declare `forward_pass` *and*
`backward_pass` *and* `update_conn` — that combination is what CBP needs and it
is already legal.

---

# Stage 0 — the non-stationary supervised harness

Everything in Stage 1–2 is gated on this. It validates each rule against a
signal we fully control, at batch 1, with no RL code — separating *"the rule is
mis-implemented"* from *"bootstrapping broke the rule"*. It is also where claim 2
gets tested in isolation.

Ship: **`examples/nonstationary.py`**, **`tests/test_nonstationary.py`**.

## The task: a drifting linear teacher

Extend `teacher_task` from `examples/dst_sparse.py`. A fixed random linear
teacher `T ∈ R^{classes × d}` produces argmax labels. Non-stationarity is a
**rotation** of the teacher applied every `switch_period` cycles:

```python
def drift(T, theta, rng):
    """Rotate the teacher toward a fresh random one by angle theta."""
    T2 = rng.standard_normal(T.shape)
    T2 /= np.linalg.norm(T2, axis=1, keepdims=True) * np.linalg.norm(T, axis=1, keepdims=True) ** -1
    return math.cos(theta) * T + math.sin(theta) * T2
```

One knob, `theta`, grades severity continuously: `theta → 0` is stationary,
`theta = π/2` is a full resample. That is the Plasticine ladder (standard →
continually varying) with a single continuous parameter instead of a scenario
list, and it means severity and frequency can be swept independently.

**Grid.** `theta ∈ {0, π/8, π/4, π/2}` × `switch_period ∈ {∞, 40, 10}` cycles.

`examples/parallel_mnist/` carries a `permute_period` input-distribution shift if
a harder, non-synthetic variant is wanted later. Do not start there — the linear
teacher's difficulty is analytically controllable and that is what Stage 0 needs.

## Metrics, recorded per cycle

- online accuracy and loss
- **recovery time** — cycles to return to the pre-switch accuracy plateau. This
  is the plasticity metric; everything else is diagnostic.
- live-edge count and realized sparsity
- dormant fraction (units with activation EMA below τ)
- mean `|w|` and mean `|w|` of edges formed since the last switch
- wall-clock per cycle

## Gates, in order

**G0 — plasticity loss must be visible in the dense baseline.** Run dense first
and confirm **recovery time grows across successive switches**. If it does not,
the task is too easy and *nothing downstream is measurable* — increase `theta`,
shorten `switch_period`, or lengthen the run until it does. **Do not proceed
past G0.** This gate exists because a method that "fixes" a problem the setup
never exhibited is an artifact.

**G1 — each rule holds its target sparsity across a switch**, within one churn's
quantile error. A rule whose density drifts on the switch is mis-implemented.

**G2 — CBP reduces recovery-time growth relative to dense** at matched density.
This is the first real result in the plan.

## Baselines in the Stage-0 table

Dense; static random sparsity at matched density (free — `from_edges` with an ER
mask and churn disabled); SET; RigL; then the four methods.

---

# Method 1 — CBP (build first)

Continual Backprop, Dohare et al., *Nature* 2024. Continually reinitialize the
lowest-**contribution-utility** mature units. Utility is

```
utility(i) = |activation(i)| * sum_{j in out(i)} |w_ij|
```

running-averaged, gated by an age threshold so fresh units are not recycled.

**Why it fits plastax exactly.** `Σ|w_out|` is a reduction over unit `i`'s
*outgoing* edges — one `BackwardPass` monoid reduction. The literature writes
this as a dense column-norm because that is what a framework offers; here it is
the native operation. **This is the single strongest demonstration in the plan**,
and it is why CBP goes first.

**Columns.** `cbp/util` (f32, running average), `cbp/age` (i32, counted in
*churns*, not steps), `cbp/reset` (f32 flag, 0/1), `cbp/thresh` (f32).

**Churn net phases.**

- `forward_pass` — per-unit incoming reduction; feeds the v1 threshold below.
- `backward_pass` — `map` returns `jnp.abs(c[px.WEIGHT, cid])` for each outgoing
  edge; `apply` writes
  `cbp/util ← (1-η)·util + η·|u[ACTIVATION,i]|·acc`, bumps `cbp/age`, and sets
  `cbp/reset = (util < thresh) & (age > maturity)`.
- `update_conn = CbpReinit` —
  - `incoming(u, dst, src, c, cid, g)`: when `u[CBP_RESET, dst]`, write a fresh
    weight from `_hash01(src, dst, u[CBP_AGE, dst])` scaled by sparse fan-in,
    **and zero this edge's optimizer state columns** — they are conn fields, so
    write them in the same `ConnWrite`. Otherwise return an empty write.
  - `outgoing(u, src, dst, c, cid, g)`: when `u[CBP_RESET, src]`, write
    `WEIGHT = 0.0` — the paper's zero-outgoing rule, which keeps the reset from
    perturbing the network's current function.
- `prune_conn = None`, `add_conn = None`. CBP does not change topology; it
  reallocates through it.
- Reset `cbp/age = 0` and `cbp/util` to the maturity-window mean for reset units.

**The threshold — build both versions.**

- **v0 (oracle).** Host reads `state.units["cbp/util"]`, takes the per-level
  quantile at replacement rate ρ, broadcasts into `cbp/thresh` exactly as
  `set_zeta` does. `O(N)` host round-trip per churn. Honest, obviously correct,
  and it is the reference v1 is checked against.
- **v1 (local, two-hop).** Each destination reduces over its **incoming** edges
  the utilities of its sources → writes a fan-in mean utility. Each source then
  reduces over its **outgoing** edges those destination means → compares its own
  utility against the average of what its neighbourhood sees. Pure local, two
  monoid reductions, zero global state — the same move that turned SET's global
  top-k into a per-unit half-normal quantile.

**Gate.** v1's per-cycle reset set matches v0's to ≥ 0.9 Jaccard on Stage 0. If
it does not, ship v0 and report v1 as an open problem. Do not fudge the
threshold to make the gate pass.

**Gate OUTCOME, measured 2026-08-31: 0.393 → 0.804 after T9. Still short of
0.9, but the failure mode is gone.** The original bar was a fraction of the
neighbourhood MEAN — a threshold on a value, which cannot express "the lowest
ρ" — and it fired 2.6–5× too many resets, capping the gate on rate alone. It is
now the analytic ρ-quantile of the neighbourhood's LOG-utility moments
(log-normal chosen by measurement: 0.6% error against 41–57% for gamma,
half-normal and exponential), and v0 carries its fractional count the way the
authors' code does. Rate now matches to 1.07×.

What remains is genuine ranking disagreement on the marginal unit, roughly one
or two of ten per churn, close to the metric's own resolution at this width. The
threshold was NOT tuned to reach 0.9. Next steps — score v0 vs v1 on recovery
time at matched rate, and re-score at a larger width — are in
[`../todo/rl-cbp-v1-rate-control.md`](../todo/rl-cbp-v1-rate-control.md).

**This is also evidence for how to answer the other three global-scalar cases**
(T9's sibling problem in `rl-layernorm-gap.md`): a parametric quantile from
local moments replaced a host round-trip here, so try that on UPGD's `eta`
before building a per-level unit-reduction phase into the library.

**Deviation to record.** The paper's utility uses batch activation statistics; at
batch 1 it is an EMA over steps. Note it, and sweep η.

**Tests** (`tests/test_cbp.py`):
1. utility equals a NumPy oracle `|act| · Σ|w_out|` on a hand-built 3-layer net;
2. a reset unit's outgoing weights are exactly zero and its optimizer state is
   zeroed after one churn;
3. a unit below the maturity age is never reset;
4. v0/v1 Jaccard gate.

**Commits.** `feat(examples): continual backprop as local unit policies`,
`test(examples): oracle-check CBP contribution utility`.

---

# Method 1b — UPGD (build alongside CBP; shares the utility machinery)

Utility-based Perturbed Gradient Descent — Elsayed & Mahmood, NeurIPS 2023
(`arXiv:2302.03281`) and ICLR 2024 (`arXiv:2404.00781`). **CBP's idea at edge
granularity instead of unit granularity**, from the same lab, and already
evaluated at batch 1 with no replay and unknown task boundaries — the exact
setting this plan targets. It was not in the original four; the streaming survey
put it there.

Per-*weight* utility is the loss change from zeroing that weight, first-order:

```
U(w) = -(dL/dw) * w
```

and the update perturbs in inverse proportion to utility:

```
w <- w - alpha * (dL/dw + xi) * (1 - Ubar),   xi ~ N(0, sigma^2)
Ubar = sigmoid(U / eta)
```

High-utility weights are protected from the noise (anti-forgetting); low-utility
weights are perturbed hard (rejuvenated plasticity).

**Why this is the tightest fit in the whole plan.** `dL/dw` for edge `(src,dst)`
is `u[GradPreAct, dst] * u[ACTIVATION, src]` — the delta-rule factorization
plastax already computes for every optimizer bundle. So `U = -(dL/dw)*w` is a
**pure per-edge local quantity**, and UPGD is *one `UpdateConn`*: no churn net,
no prune, no add, no extra pass. It drops into `optim/` shape directly.

**Columns.** One f32 per-edge running utility (`upgd/util`), living in
`state_fields` exactly like adam's moments.

**The one global piece.** `eta` is the maximum weight utility across the
network. Same shape of problem as CBP's per-layer ranking, and the same two
answers: **v0** host-side max broadcast into a column (`set_zeta` pattern), and
**v1** a local surrogate — a per-unit running max over incoming edges, read by
each edge from its destination. Check v1 against v0 the same way.

**Why build it with CBP.** They are the same construct at two granularities, so
they share the utility EMA, the v0/v1 threshold pattern, and the tests. Having
both lets the study ask a question the literature has not: **at matched
intervention budget, is capacity better reallocated per-unit (CBP) or per-edge
(UPGD)?** In a mask-based framework that question is awkward; on an edge arena
it is natural.

**Baselines UPGD was measured against** (reuse for Stage 0): SGDW, AdamW, PGD,
Shrink & Perturb, and streaming EWC / SI / MAS / RWalk, on input-permuted MNIST
and label-permuted EMNIST/CIFAR-10/mini-ImageNet. Note the overlap with
`examples/parallel_mnist/` — the permuted-MNIST harness in this tree is the same
family of task, which makes the Stage-0 fallback a stronger option than it first
looked.

**Test** (`tests/test_upgd.py`): utility matches a NumPy oracle `-(dL/dw)*w` on a
hand-built net; a zero-utility edge receives the full perturbation and a
max-utility edge receives none; v0/v1 agreement.

**Commits.** `feat(examples): utility-based perturbed gradient descent`,
`test(examples): oracle-check UPGD per-edge utility`.

---

# Method 2 — GMP

Gradual magnitude pruning, Zhu–Gupta cubic schedule. Dense → sparse, no growth.
Obando-Ceron et al., ICML 2024. Land this alongside CBP; it is ~30 lines.

**Traits.** Churn net = `AnnealedMagnitudeStats` (verbatim from `xmc.py`) +
`SetPrune` (verbatim from `dst_sparse.py`). `add_conn = None`. The method *is* a
scheduling change on shipped machinery.

**The schedule.** `AnnealedMagnitudeStats` prunes a fraction ζ of *currently
live* edges; GMP specifies an *absolute cumulative* sparsity `s_t`. Convert
host-side, then broadcast with `set_zeta`:

```python
s_t    = s_f * (1.0 - (1.0 - t / T) ** 3)           # Zhu & Gupta cubic
zeta_t = (s_t - s_prev) / max(1e-8, 1.0 - s_prev)   # fraction of the live set
```

**Build.** Start the arena dense. Expect the slowest wall-clock in the study —
that is a measurement, not a bug; record it.

**Test** (`tests/test_gmp.py`): realized live-edge count tracks `s_t` to within
one churn's quantile error at `s_f ∈ {0.75, 0.90, 0.95}`.

**Commits.** `feat(examples): gradual magnitude pruning schedule`,
`test(examples): pin GMP sparsity trajectory`.

---

# Method 3 — NE

Liu et al., ICLR 2025. Sparse → dense growth with dormancy pruning. The closest
structural match to what plastax uniquely supports, and the only method here
framed as *plasticity* rather than compression.

**Build.** ER-sparse at 25 % of target live edges, with `capacity_headroom`
sized for the terminal density so growth never overflows.

**(a) Elastic topology generation.** `𝕀_grow = ArgTopk_{i∉θ}(|∇θ L|)` is
`GradientGrow.score` **verbatim**. What is new is the cosine-annealed growth
*rate* `α/2·(1 + cos(tπ/T))`.

`max_candidates` is a static trace-time bound, so the count cannot be a traced
value. Thin the candidates stochastically instead:

```python
def score(self, u, src, dst, g):
    deeper = u[px.LEVEL, dst] > u[px.LEVEL, src]
    keep   = _hash01(src, dst, u[SET_CURSOR, dst]) < u[NE_RATE, dst]
    grad   = jnp.abs(u[GradPreAct, dst] * u[px.ACTIVATION, src])
    return jnp.where(deeper & keep, grad, -jnp.inf)
```

`NE_RATE` is a per-unit column written host-side each cycle from the cosine
schedule — the `set_zeta` pattern again. Expected growth count ∝ rate; exact in
expectation, local, no new machinery. `-inf` is the hard veto (see
`fix(phases)`); keep it.

**(b) Dormant neuron pruning.** The paper's `𝕀_prune = {i | f(θ_i) = 0}` is a
batch statistic. At batch 1 use an EMA: an `ne/act_ema` column updated in the
**train** forward pass, `ema ← (1-β)·ema + β·|activation|`. The churn net's
`prune_conn` tombstones every incoming edge of a dormant destination:

```python
def predicate(self, u, c, cid, g):
    dst = px.UnitIdx(c[px.TO_ID, cid])
    return u[NE_ACT_EMA, dst] < u[NE_DORMANT_TAU, dst]
```

Conn-local. This column also supplies the dormant-fraction metric for every
other method in the study, so land it early.

**(c) Deviations to record, not paper over.**

- The `ω·|𝕀_grow|` per-layer prune cap is a global count. Drop it in v1; if
  dormancy pruning runs away, reintroduce it as a per-unit probability, never a
  count.
- **Experience-review consolidation assumes a replay buffer.** Streaming RL has
  none. Drop it; report as "NE without consolidation". If the plasticity–
  stability tradeoff visibly bites, that is itself a result about streaming + NE.
- **Terminal density (decision 5).** Run both: uncapped (faithful — ends dense,
  at which point plastax has no representational advantage left) and capped at a
  `density_target` (e.g. 10 %). Report both trajectories.

**Tests** (`tests/test_ne.py`): live-edge count follows the cosine growth
envelope within tolerance; a forced-dormant unit loses all incoming edges in one
churn; growth never overflows (`assert result.overflow == False`).

**Commits.** `feat(examples): neuroplastic expansion growth and dormancy`,
`test(examples): pin NE growth envelope`.

---

# Method 4 — RLx2-λ

Tan et al., ICLR 2023. Three components: RigL topology evolution, multi-step TD
targets, dynamic-capacity replay buffer.

**Component 1 is already shipped.** `GradientGrow` + `SetPrune` in
`examples/dst_sparse.py` *is* RLx2's topology half, unchanged.

**Components 2 and 3 under streaming.**

- *multi-step TD target* → **λ-returns via eligibility traces**. RLx2's
  multi-step target is the batch-mode answer to a specific problem: the
  value-estimation error a sparse critic accumulates under bootstrapping. The
  eligibility trace is the streaming answer to the same problem. This is a
  principled substitution, not a convenience.
- *dynamic-capacity replay buffer* → **no analogue.** Report as dropped.

The deliverable is **RLx2-λ: RigL topology on a stream-AC(λ) critic.** Label it
that way in every table; never call it RLx2.

**Success criterion (decision 4).** Not "matches the paper's numbers" — the
setting is different. The adaptation succeeds if it reproduces RLx2's *central
mechanism*:

> at matched density, RigL topology evolution on the critic degrades at λ = 0
> and recovers as λ increases.

That is the streaming restatement of RLx2's multi-step-target ablation. If the λ
sweep shows it, the claim "non-stationary bootstrapping is what makes sparse RL
hard, and a longer-horizon target is what fixes it" survives the move to batch 1,
and RLx2-λ becomes the representative example of the family. If the sweep is
flat, that is equally publishable and must be reported as such.

**Config note.** Disable bias terms in the sparse nets — reported to
substantially improve RigL at high sparsity. Make it a flag, set it on by
default, and record the ablation.

**Module split.** Per `arXiv:2510.12096` (MST): DST on the **critic**, static
sparse on the **actor** (dynamic rewiring destabilizes policy optimization),
dense encoder where one exists. Run the swapped ablation too — it is cheap and it
is the current frontier's headline claim.

---

# Stage 1 — the streaming RL harness

**`examples/streaming_rl/`** — new package, mirroring `examples/parallel_mnist/`.

1. **`obgd.py` — ObGD as a plastax optimizer bundle.** Implements the
   `optim.Optimizer` protocol: `state_fields` carries the per-edge eligibility
   trace (and the adaptive variant's second moment); `update_conn()` returns the
   overshooting-bounded step. Per-edge traces are exactly what `state_fields` was
   built for — see `optim/_adam.py` for the shape. Port from
   `github.com/mohmdelsayed/streaming-drl/optim.py`; read the paper's equations,
   do not guess the bound. Per decision 6, build here; promote to
   `src/plastax/optim/` (scope `optim`) only if it comes out clean.
   The published rule is `M = alpha*kappa*delta_bar*||z_w||_1`, then
   `alpha <- min(alpha/M, alpha)`, then `w <- w + alpha*delta*z_w`, with
   `delta_bar = max(|delta|, 1)` and `kappa > 1`.
   *Optional successor:* Intentional TD(λ) (`arXiv:2604.19033`) beats ObGD and is
   also per-edge (traces + RMSProp-style diagonal scaling), so it is a second
   bundle of the same shape. Do not build it until ObGD passes the bring-up gate.
2. **`env.py`** — Gymnasium MuJoCo. Host-side stepping is fine; the env is not
   the bottleneck at batch 1. CartPole/Acrobot for bring-up.
3. **`agent.py`** — stream-AC(λ) and stream-Q(λ) over plastax nets: actor and
   critic each a `Network` subclass with train/churn/eval modes, plus LayerNorm
   and observation/reward scaling per the paper.
4. **`run.py`** — the cycle loop, mirroring `dst_sparse.py:run`: N env steps,
   then one churn step, then metrics.

**Bring-up gate.** Dense plastax stream-Q solves CartPole-v1 and matches the
reference implementation's learning curve **before any rewiring is enabled**. If
that fails, no rewiring result downstream means anything. Do not proceed.

---

# Stage 2 — the comparison

**Baselines in every table.** Dense stream-x (SparseInit disabled, LeCun init);
**stream-x as published — SparseInit at s = 0.9, static, never rewired.** That
second one is the null hypothesis for this entire study: it is the configuration
the streaming RL literature actually runs, it is free (`from_edges` with an ER
mask and churn disabled), and every rewiring result must clear it. Then SET,
RigL, and the five methods.

**Metrics per cycle.** Return; live-edge count and realized sparsity; dormant
fraction; mean `|w|`; wall-clock per step. Carry Stage 0's recovery-time metric
wherever the environment has an identifiable distribution shift.

**Non-stationarity axis.** Streaming RL is non-stationary by construction. For a
graded axis, take Plasticine's scenario ladder
(`github.com/RLE-Foundation/Plasticine`) rather than inventing one — matching
their metrics is what makes the numbers legible to this community.

---

## Sequencing

```
Stage 0   examples/nonstationary.py — drifting-teacher harness
          ├─ G0  dense shows growing recovery time      ← HARD GATE
          ├─ CBP v0   (host quantile oracle)
          ├─ CBP v1   (two-hop local threshold)         ← gate: ≥0.9 Jaccard vs v0
          ├─ UPGD v0/v1 (per-edge utility; shares CBP's machinery)
          ├─ G2  CBP beats dense on recovery time       ← first real result
          ├─ GMP      (schedule only, reuses xmc.py)
          └─ NE       (growth anneal + dormancy; ne/act_ema feeds every method's metrics)
Stage 1   ObGD bundle → dense stream-Q on CartPole      ← gate: matches reference
Stage 2   four methods × stream-AC(λ) × MuJoCo/DMC, MST module split
          └─ RLx2-λ success criterion: λ sweep reproduces the multi-step effect
```

Stage 0 is strictly ordered before Stage 1: every rule must be correct against a
signal we control before it meets a bootstrapped one. Within Stage 0, G0 is
strictly ordered before everything else.

---

## Risks

- **G0 fails** — the synthetic teacher never exhibits plasticity loss. Fall back
  to `examples/parallel_mnist/`'s permuted-MNIST shift, which is a harder,
  non-synthetic distribution change and has prior results in this tree.
- **CBP v1 does not match v0.** Ship v0, report v1 open. The local-threshold
  question is worth stating even unresolved.
- **RigL's signal degrades under bootstrapping** (Graesser et al.). Expected;
  it is claim 2. The Stage-0 → Stage-2 delta is the measurement.
- **Wall-clock loses to dense** at these widths. Known and quantified
  (`project_plastax_perf_ceiling`). Report it beside the capability result.
- **ObGD port is subtly wrong.** The bring-up gate against the reference
  implementation's CartPole curve is what catches this; do not skip it.

---

# Open questions: streaming RL × structural search

Surveyed 2026-08-30, after the landscape survey above. Each question below is
open in the literature, answerable with the machinery this plan already builds,
and carries a concrete task. Tasks are additive to the staging above; none of
them block Stage 0.

## Where "structural search" currently sits in RL

Two granularities exist, and there is nothing in between or below.

- **Layer granularity.** *An Integrated Approach to Neural Architecture Search
  for Deep Q-Networks* (Rahmani, Yazdannik, Tayefi & Roshanian, `2510.19872`)
  puts a NAS controller inside the DQN loop, resampling from 27 configurations
  (depth ∈ {2,3,4} × width ∈ {32,64,128} × activation) every 200 episodes,
  transferring matching weights and pruning replay to 25 %. Their claim is
  strong — architecture adaptation is *necessary* for sample efficiency in
  online DRL, not merely helpful — but the evidence is one environment
  (Inverted Pendulum), three seeds. Treat the claim as a hypothesis worth
  testing, not a result to build on. It uses a replay buffer.
- **Edge granularity.** Everything in the DST column of the landscape survey.
  All batched.

**Nothing operates at either granularity in the streaming setting**, and nothing
compares granularities. AutoRL@RLC 2026's twelve accepted papers contain no
architecture-search work at all, despite "neural architecture design and search
for RL" being an advertised topic — the topic is called for and unserved.

## Q1 — Does gradient-guided regrowth survive a single-sample estimate?

**The question.** RigL's growth score for an absent edge is
`|grad_pre_act[dst] · activation[src]|`. Under replay that is a batch mean over
256 transitions. At batch 1 it is a single-sample estimate of the same quantity,
with the variance that implies. Graesser et al. already found this signal weak
in DRL *with* replay averaging it. Removing the average should make it worse.

**The obvious fix, and it is plastax-native.** stream-x fixed the *update* rule
by putting an eligibility trace on it. Do the same to the *growth score*.

An absent edge has no per-edge state column to trace — but it does not need one.
Trace the two per-unit factors separately:

```
z_grad[dst] <- gamma*lam*z_grad[dst] + grad_pre_act[dst]     # one f32 column
z_act[src]  <- gamma*lam*z_act[src]  + activation[src]       # one f32 column
score(src, dst) = |z_grad[dst] * z_act[src]|
```

Two `O(N)` per-unit trace columns yield a smoothed score for all `O(N²)` absent
edges, because the delta-rule factorization commutes with the trace. This is the
same move that makes RigL local in `dst_sparse.py`, applied one level up. In a
mask framework you would have to materialize and trace a dense `N × N` gradient
to get the same thing.

**Task T1.** Add `method="traced_rigl"` to the growth switch: two trace columns,
`lam` swept alongside the critic's own λ. Compare against untraced RigL at
matched density in Stage 0 first, then Stage 2. If untraced RigL degrades at
batch 1 and traced RigL recovers it, that is the plan's second real result and
it is a new algorithm, not a port.

## Q2 — What is the right timescale for structural change?

**The question.** Batched DST rewires every `ΔT` gradient steps, each seeing 256
samples. Streaming sees one. Should the churn period be matched in *samples* or
in *updates*? The two differ by 256×, and every published `ΔT` was tuned in the
batched regime — so both readings are unjustified defaults.

**Task T2.** Sweep churn period over three orders of magnitude in Stage 0, where
it costs minutes. Report the optimum in both units. This is cheap, nobody has
done it, and every subsequent experiment depends on getting it roughly right.

## Q3 — Layer, unit, or edge?

**The question.** Given a fixed intervention budget — some number of weights
allowed to change identity per unit time — is capacity better reallocated by
resizing layers (NAS-DQN), resetting units (CBP, ReDo), or rewiring edges
(SET, RigL, UPGD)? The three literatures do not cite each other and no framework
holds all three.

**Why plastax can ask it.** All three are traits over one arena: layer resizing
is bucket capacity, unit reset is `UpdateConn` gated on a unit flag, edge
rewiring is `prune_conn`/`add_conn`. Same state, same step, same measurement.

**Task T3.** With CBP (unit) and UPGD (edge) both landed per Method 1/1b, define
a matched budget — weights whose value is set by something other than the
gradient, per 1000 steps — and run the granularity sweep. Layer-level is
optional and can be a coarse two-point check (narrow vs wide) rather than a NAS
controller.

## Q4 — Is online sparsity instability a property of being online, or of missing stabilizers?

**The question.** Sokar et al. found SET destabilized past 50 % sparsity in
online RL. stream-x runs stably at 90 % — but statically, and with LayerNorm,
ObGD, traces, and reward/observation scaling that Sokar's setup lacked. So the
50 % ceiling may be a fact about online rewiring, or an artifact of a 2022
training stack. These have opposite consequences for this whole plan.

**Task T4.** Ablate the stream-x stabilizer stack under active rewiring at 90 %
sparsity: LayerNorm on/off, ObGD vs plain SGD, traces on/off, scaling on/off.
Run it as soon as Stage 1's bring-up gate passes, before the full method sweep —
if rewiring at 90 % is unstable regardless, the study needs to move to a lower
density and it is much cheaper to learn that early.

## Q5 — Substitute or complement?

**The question.** stream-x's stability comes from a stack: SparseInit, LayerNorm,
ObGD, traces, scaling. Weight clipping (Elsayed, Lan, Lyle & Mahmood, RLC 2024,
`2407.01704`) is a further member of the same family. If structural plasticity
subsumes any of these, that is a simplification result; if it is orthogonal, an
additive one. Nobody has run the ablation because nobody has the rewiring.

**Task T5.** Falls out of T4's ablation grid at no extra cost — report it as its
own question. Add weight clipping as a cheap sixth arm; it is a bounded
`ConnWrite` on the weight column and costs almost nothing to implement.

## Q6 — Partial observability, memory, and where the capacity should go

**The question, and the most serious caution in this document.** *Forager*
(Tang, Xiong, Hakhverdyan, Patterson, Adkins, He, Elelimy, Mohammad Panahi,
White & White, `2605.01131`) is a lightweight partially-observable continual RL
testbed with a constant memory footprint and a task-stream variant. Its finding:
agents do exhibit plasticity loss and mitigations do help, **but state
construction helps most.** It also notes that CRL work has rarely considered
partial observability, memory or recurrence at all.

If state construction dominates, then a rewiring intervention applied to a value
head is second-order, and a plan that only rewires the critic is measuring the
wrong thing. The interesting version of the question is whether topology change
helps *state construction* — the recurrent, representational part — rather than
the value head.

**Why this is unusually tractable here.** plastax already carries recurrent
machinery that most sparse-training frameworks do not:
`examples/reservoir.py`, `examples/echo_state_network.py`,
`examples/sharded_reservoir.py`, `Propagation.PIPELINE`, and
`tests/test_recurrent_pipeline.py`. Streaming + recurrence + rewiring is
completely untouched territory, and the pieces are already in the tree.

**Task T6.** After Stage 2, evaluate on Forager rather than only MuJoCo — it is
lightweight, constant-memory, and explicitly built for this question. Rewire the
recurrent/state-construction network, not just the value head, and report the
split. Treat this as the strongest follow-on, not a stretch goal.

## Q7 — Does the module-specific finding even apply?

**The question.** MST (`2510.12096`) prescribes dense encoder, DST critic, static
sparse actor. Streaming agents at 2×128 do not have a separable encoder — the
same small network does representation and value. So either the prescription
degenerates, or "encoder" maps onto early layers and the prescription becomes a
per-*level* policy rather than a per-module one.

**Task T7.** plastax's per-level bucketing makes the second reading directly
expressible — `shortlist_per_level` already partitions candidates by level.
Test whether an early-levels-dense / late-levels-DST split reproduces MST's
module effect in a network with no module boundaries. Cheap, and it is a
sharper claim than MST's if it holds.

## Q8 — Evaluation variance

*Performance Variation in Deep Reinforcement Learning* (Tanaka & Mahmood,
AutoRL@RLC 2026) addresses exactly the regime we are in. Streaming results are
high-variance and our effect sizes may be small.

**Task T8.** Fix the seed count and the reported statistic **before** running
Stage 2, not after seeing the results. Read that paper first and adopt its
protocol.

---

## Task summary

Each task has a scoped file in `../todo/`, following that directory's
one-topic-per-file convention. Those files carry the experiment design; this
table is the index.

| Task | Question | Stage | Cost | File |
| --- | --- | --- | --- | --- |
| T1 | traced growth score (new algorithm) | Stage 0, then 2 | low — two unit columns | [rl-traced-growth-score.md](../todo/rl-traced-growth-score.md) |
| T2 | churn timescale: samples or updates | Stage 0 | low | [rl-churn-timescale.md](../todo/rl-churn-timescale.md) |
| T3 | layer vs unit vs edge at matched budget | Stage 0, after 1/1b | medium | [rl-granularity-budget.md](../todo/rl-granularity-budget.md) |
| T4 | is 90 % rewiring stable at all | right after Stage 1 gate | low, **run early** | [rl-sparsity-stability.md](../todo/rl-sparsity-stability.md) |
| T5 | rewiring vs the stabilizer stack | rides on T4 | ~free | [rl-stabilizer-ablation.md](../todo/rl-stabilizer-ablation.md) |
| T6 | Forager, recurrence, state construction | after Stage 2 | high, highest value | [rl-forager-recurrence.md](../todo/rl-forager-recurrence.md) |
| T7 | per-level vs per-module sparsity policy | Stage 2 | low | [rl-per-level-policy.md](../todo/rl-per-level-policy.md) |
| T8 | fix the evaluation protocol first | before Stage 2 | ~free | [rl-eval-protocol.md](../todo/rl-eval-protocol.md) |
| T9 | rate-control CBP's local threshold — **largely done**, 0.393 → 0.804 | Stage 0 | low | [rl-cbp-v1-rate-control.md](../todo/rl-cbp-v1-rate-control.md) |

**T1 and T4 are the two that change what gets built.** T1 is the plan's most
likely novel contribution; T4 decides whether the target density is viable at
all. Do T4 the moment stream-Q clears CartPole.

## Community contacts

Both relevant RLC 2026 workshops are organized by people directly in this line
of work, which matters for the outreach the user is planning.

- **Continual RL Workshop @ RLC 2026** — Ziyan Luo (McGill), Xutong Zhao
  (Polytechnique), Tianwei Ni (UdeM), Nishanth Anand (McGill),
  **Guozheng Ma** (NTU), **Anna Hakhverdyan** (Brown). Ma is first author of
  *Network Sparsity Unlocks Scaling* (ICML 2025) and *Rethinking DST for
  Scalable DRL* (`2510.12096`), and a Plasticine author — the two papers this
  plan's protocol is built on. Hakhverdyan is a Forager author.
  Accepted-papers page was not yet populated as of 2026-08-30 (camera-ready
  pending).
- **AutoRL @ RLC 2026** — Theresa Eimer (Hannover), Julian Dierkes (Mila/RWTH),
  **Johan Obando-Ceron** (Mila/UdeM), Raghu Rajan and André Biedenkapp
  (Freiburg). Obando-Ceron is first author of *a pruned network is a good
  network* (ICML 2024) and the MoE scaling work.
- **Streaming line** — Elsayed, Vasan, Sharifnassab, De Asis, Mahmood, Sutton
  (Alberta); Forager is Martha White and Adam White's group, same place.
