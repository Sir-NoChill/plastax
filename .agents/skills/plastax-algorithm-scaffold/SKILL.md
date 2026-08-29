---
name: plastax-algorithm-scaffold
description: >-
  Scaffold a new contribution to plastax against the exact contract it must
  satisfy — a plasticity algorithm (forward/backward pass, loss, connection
  update, prune or grow policy), an optimizer bundle, a topology generator, a
  named monoid, or (rarely) a whole new phase category. Use this whenever you
  are adding a learning rule, plasticity rule, SET/RigL-style dynamic-sparse
  policy, Hebbian/eligibility-trace update, a new optimizer (sgd/adam-like), a
  new layer/topology generator, a new reduction monoid, or wiring any of these
  into a Network subclass. It gives the Protocol signatures, which files to
  touch, a code skeleton, the reference example to copy the idiom from, and the
  test + docs + commit checklist. Prefer this over improvising the shape of a
  new trait. Load plastax-architecture first if unsure where the change goes.
---

# plastax algorithm scaffold

Add a new algorithm to plastax against its real contract, so it slots into the
declarative trait system, respects the invariants, and ships with the tests,
docs, and commit the repo requires.

**Before writing code**, make sure you have the layout and invariants: read
[`AGENTS.md`](../../../AGENTS.md) and [`.agents/architecture.md`](../../architecture.md)
(or invoke the `plastax-architecture` skill). The single most important fact:

> **A new plasticity algorithm is almost always just a new policy class
> implementing an existing Protocol, assigned as a class attribute on a
> `Network` subclass. It touches NO framework module.** You only edit
> `traits.py`/`phases.py`/`sweep.py` if you need a phase *category* that does
> not already exist.

## Step 0 — Classify the contribution

Pick the row; it decides everything downstream.

| You are adding… | Kind | Files you touch | Skeleton |
|---|---|---|---|
| A forward/backward pass, loss, connection-update rule, prune policy, or grow policy | **Policy** (existing Protocol) | a new class in your module/example; assign on a `Network` subclass | §A |
| An optimizer (sgd/adam-like weight update) | **Optimizer bundle** | `optim/_<name>.py` + `optim/__init__.py` | §B |
| A layer / connectivity generator | **Topology generator** | `topology.py` | §C |
| A new reduction for accumulators | **Named monoid** | `monoid.py` | §D |
| A phase category beyond the seven | **New phase** (rare, invasive) | `traits.py` → `phases.py` → maybe `sweep.py` | §E |

Confirm the classification and the commit **scope** (`SCOPES.md`) with the user
before implementing if there is any ambiguity.

## The universal contract (applies to every kind)

1. **Policies are pure per-element functions.** They receive `UnitView`/
   `ConnView` for one element and return `UnitWrite`/`ConnWrite`. Never index
   raw columns, never read another element's state, never keep state on the
   instance beyond hyperparameters. (invariant #6)
2. **Reads:** `u[SPEC, idx]` / `c[SPEC, cid]` → a scalar. **Writes:**
   `UnitWrite.of((SPEC, value), …)` / `ConnWrite.of((SPEC, value), …)`. An empty
   `ConnWrite.of()` is a valid no-op.
3. **Shapes stay static; state stays donatable.** Do not introduce a data-
   dependent shape or a leaf that changes shape/dtype across a step. (invariants
   #2, #5)
4. **Extra fields** your policy needs are declared on the `Network` via
   `extra_unit_fields` / `extra_conn_fields` (tuples of `FieldSpec`); they must
   not collide with reserved names (`from_id`, `to_id`, `dead`, `weight`,
   `activation`, `level`). A per-connection field defaults to its
   `FieldSpec.default` on a regrown edge — rely on that for zero-init.
5. **Copy the idiom from working code**, don't invent it. The canonical trait
   implementations live in `examples/mlp_xor.py` (forward/backward/loss/update),
   `examples/dst_sparse.py` (prune/grow, SET & RigL), and `optim/_adam.py`
   (a stateful update). Read the relevant one before filling a skeleton.

---

## §A — A policy implementing an existing Protocol

Signatures (from `traits.py`; `Acc` is your accumulator type, `GS` the network
globals type — use `object` if you never read globals):

```python
# ForwardPass[Acc, GS]: accumulate over incoming edges into the DESTINATION unit
class MyForward:
    combine: MonoidTree = px.monoid.sum_        # or a (tuple/dict) product of monoids
    def map(self, u, dst, src, c, cid, g) -> Acc: ...      # per-edge contribution
    def apply(self, u, i, g, acc) -> UnitWrite: ...        # finalize unit i from acc

# BackwardPass[Acc, GS]: same shape, but accumulates into the SOURCE unit
class MyBackward:
    combine: MonoidTree = px.monoid.sum_
    def map(self, u, dst, src, c, cid, g) -> Acc: ...
    def apply(self, u, i, g, acc) -> UnitWrite: ...

# Loss[GS]
class MyLoss:
    def per_output(self, u, i, target, g) -> tuple[Float[Array, ""], UnitWrite]: ...

# UpdateConn[GS]: two-pass connection state update
class MyUpdate:
    def incoming(self, u, dst, src, c, cid, g) -> ConnWrite: ...   # all incoming writes land first
    def outgoing(self, u, src, dst, c, cid, g) -> ConnWrite: ...   # then all outgoing (often no-op)

# PruneConn[GS]: tombstone by predicate
class MyPrune:
    def predicate(self, u, c, cid, g) -> Bool[Array, ""]: ...      # True ⇒ prune this edge

# AddConn[GS]: K-bounded growth
class MyGrow:
    max_candidates: int = ...
    def score(self, u, src, dst, g) -> Float[Array, ""]: ...   # -inf = HARD VETO (never grown)
    def init(self, u, src, dst, g) -> ConnWrite: ...           # new edge's fields (WEIGHT + yours only)
    # optional shortlist to avoid the O(num_units^2) grid:
    # max_candidate_units: int; def importance(self, u, i, g) -> Float[Array, ""]: ...
    # shortlist_per_level: bool = True

# ResetGlobal[GS]
class MyReset:
    def reset(self, g) -> GS: ...
```

Wire it onto a network:

```python
class Net(px.Network[GS]):
    forward_pass = MyForward()          # required; the rest are optional
    backward_pass = MyBackward()
    loss = MyLoss()
    update_conn = MyUpdate()
    prune_conn = MyPrune()
    add_conn = MyGrow()
    extra_unit_fields = (MY_UNIT_FIELD,)     # FieldSpecs your policies read/write
    extra_conn_fields = (MY_CONN_FIELD,)
    propagation = px.Propagation.TOPOLOGICAL # or PIPELINE for recurrent nets
```

`__init_subclass__` validates the wiring at class-definition time — a missing
required method or a reserved-name collision fails immediately.

Direction & ordering facts to get right:
- Forward `map` accumulates into `dst`; backward `map` accumulates into `src`.
  Both `map` signatures are `(u, dst, src, c, cid, g)`.
- Phase order is fixed: forward → loss → backward → update_conn → prune_conn →
  add_conn → reset_global. So an `UpdateConn` can read what `backward` wrote
  (e.g. `grad_pre_act`), and `prune_conn` sees `update_conn`'s fresh weights.
- `UpdateConn` runs *all* incoming writes across every bucket before *any*
  outgoing pass, so the two sub-passes never race.
- A `-inf` `AddConn.score` is a hard veto, distinct from a low finite score.
  For dynamic-sparse (SET/RigL), read `examples/dst_sparse.py`: SET and RigL are
  the same class differing only in `score` (random hash vs delta-rule gradient).

---

## §B — An optimizer bundle (`optim/`)

An optimizer is a trait bundle satisfying the `Optimizer` Protocol
(`optim/__init__.py:47`), not a special object. Read `optim/_adam.py` (stateful,
three columns) or `optim/_momentum.py` (one column) as the template, then:

```python
# optim/_myopt.py
import numpy as np
from plastax._types import ACTIVATION, WEIGHT, ConnIdx, FieldSpec, UnitIdx
from plastax.views import ConnView, ConnWrite, UnitView

_STATE = FieldSpec.float32("opt/mystate")     # namespace every state column "opt/..."; default 0.0

@dataclass(frozen=True)
class _MyOptUpdateConn:                        # implements UpdateConn[object]
    lr: float
    grad_field: FieldSpec[np.float32]
    state: FieldSpec[np.float32]
    def incoming(self, u, dst, src, c, cid, g) -> ConnWrite:
        del g
        grad = u[self.grad_field, dst] * u[ACTIVATION, src]   # delta rule (exact for weighted-sum)
        # ... update state, compute new weight ...
        return ConnWrite.of((WEIGHT, new_w), (self.state, new_state))
    def outgoing(self, u, src, dst, c, cid, g) -> ConnWrite:
        del u, src, dst, c, cid, g
        return ConnWrite.of()                  # no-op: work happens in incoming

@dataclass(frozen=True)
class MyOpt:                                   # the bundle (implements Optimizer)
    lr: float
    grad_field: FieldSpec[np.float32]
    state_fields: tuple[FieldSpec[np.generic], ...] = (_STATE,)
    needs_step_counter: bool = False           # keep False; True is the untrodden globals path
    def update_conn(self) -> _MyOptUpdateConn:
        return _MyOptUpdateConn(self.lr, self.grad_field, _STATE)

def myopt(lr: float, grad_field: FieldSpec[np.float32]) -> MyOpt:   # factory
    return MyOpt(lr=lr, grad_field=grad_field)
```

Then register in `optim/__init__.py`: `from ._myopt import MyOpt, myopt` and add
both to `__all__`. Contract points:
- Gradient is the **delta rule** `grad_field[dst] * ACTIVATION[src]` unless you
  genuinely need a different source.
- `g` is `object` and must be `del`'d — never read globals (keeps one instance
  valid for every `GS`). Only set `needs_step_counter=True` if you truly need a
  globals-carried counter, and flag it — no shipped optimizer exercises that
  path.
- State columns default 0.0 so a regrown edge starts at rest (RigL/SET
  semantics) with zero work from the growth policy — do not special-case
  regrow-init in the optimizer.
- A `Network` using it sets `extra_conn_fields = opt.state_fields` and
  `update_conn = opt.update_conn()`.

---

## §C — A topology generator (`topology.py`)

Return a `Block`: an object with a `num_units: int` attribute and an
`edges(key, offset_in, offset_out) -> EdgeSet` method. Read `dense`/`conv2d` in
`topology.py` first.

```python
def my_layer(n_in: int, n_out: int, *, init: Initializer = _GLOROT_UNIFORM) -> Block:
    def make_edges(key, offset_in, offset_out) -> EdgeSet:
        # build int32 from_ids/to_ids USING the passed global offsets (never local 0-based ids),
        # and a float32 weights array of the same length; all host-side numpy/jax.
        return EdgeSet(from_ids=..., to_ids=..., weights=...)
    return _EdgeBlock(num_units=n_out, _make_edges=make_edges)
```

Contract points:
- **Host-side only** — pure numpy/jax array construction, no `NetworkState`, no
  jit, no per-step logic. It runs once at build time.
- Honor `offset_in`/`offset_out` — `sequential` composes blocks by concatenating
  id spaces; your block must not assume its ids start at 0.
- You set **initial** weights only (via the `Initializer`); ongoing updates are
  the optimizer/`UpdateConn`'s job — do not bake a learning rule in here.
- Compose via `topology.sequential(input_units(n), my_layer(...), …)` and pass
  to `NetworkBuilder.from_topology`.

---

## §D — A named monoid (`monoid.py`)

Only if `sum/prod/max/min` genuinely don't cover the accumulation. Extend the
`_Named` enum and each of the reducer / identity / pairwise / (if a JAX
collective exists) collective tables consistently, and export a prebuilt
instance. **Do not** un-guard the generic `(op, identity)` path — it must keep
raising `UnsupportedMonoidError` in v1 (invariant #8). A struct accumulator is a
`dict`/`tuple` of monoids (a `MonoidTree`) — usually you want that, not a new
named monoid.

---

## §E — A new phase category (rare, invasive)

Only when the algorithm needs a phase that is not one of the seven. In order:
1. `traits.py`: add a `@runtime_checkable` Protocol + an optional `Network`
   class attribute; extend `_validate_traits`.
2. `phases.py`: add `_build_<name>_phase` and wire it into `build_phases`'s
   ordered list — **decide its position** relative to the existing seven and
   justify it (it changes what later phases can read).
3. `sweep.py`: add the low-level gather/reduce/apply helper if the phase needs
   one not already there.
4. `step.py`/`topo.py` need no change (generic over the phase list / static).

This is a design change: write it up (IMPLEMENTATION_PLAN.md) and get the
ordering sanctioned before implementing.

---

## Step N — Tests, docs, commit (every contribution)

Do not consider the change done until:

1. **Tests** in `tests/test_<topic>.py`:
   - The right oracle tolerance for the comparison (external oracle
     `1e-4/1e-5`; internal same-order `1e-6/1e-6`; cross-mode `1e-5/1e-5`; exact
     invariant `atol=0.0`) — and a comment saying why.
   - An optimizer ⇒ an optax-parity test marked `@pytest.mark.slow` +
     `pytest.importorskip("optax")` (see `test_optim.py`), and, if it carries
     state, a regrow-zeroing test (see `test_optim_sparse.py`).
   - Ask of each test: *if the code broke, would this fail?* Seed every RNG;
     don't override `conftest.py`'s CPU/device setup.
2. **An example** (optional but preferred) in `examples/<name>.py` demonstrating
   one clear point, with a `main()` that asserts its own success criteria so it
   doubles as an acceptance test. Reuse `mlp_xor.py`'s shared traits.
3. **Docs**: update `docs/` where relevant (a new optimizer → `docs/optimizers.md`;
   a new public symbol → it must be in `src/plastax/__init__.py`'s `__all__` and
   thus `docs/api.md`). Update `.agents/architecture.md` if you changed a
   contract or the routing. Docs-in-sync is a review condition.
4. **Types & lint**: `uv run mypy --strict src` clean; `uv run ruff check` +
   `ruff format` clean; Google-style docstrings pass pydoclint. Keep `FieldSpec`
   generics intact.
5. **Review**: run the **`plastax-review`** skill.
6. **Commit** via the `agent-commit` skill: `type(scope): subject` with the
   scope from Step 0. A new optimizer is `feat(optim): …`; a topology generator
   `feat(topology): …`; a public-surface break is `feat(scope)!:` + a
   `BREAKING CHANGE:` footer + an IMPLEMENTATION_PLAN.md Deviations entry in the
   same commit. Let the hooks run; never `--no-verify`.

## Guardrails (things that look right but aren't)

- Reaching for `lax.cond` to skip a phase or a dead element — use elision / the
  null-slot trick instead (invariants #1, #4).
- Storing a live-edge count, or returning a state leaf whose shape depends on
  live count (invariants #3, #5).
- Reading `g` in an optimizer, or making `state_fields` non-zero-default and
  then special-casing regrow (§B).
- A topology generator that assumes local 0-based ids or sets a learning rule
  (§C).
- Putting orchestration (bucket loop, retrace) inside a policy — that lives in
  `phases.py`/`driver.py`, not in your class.
- Adding to `__all__` without a docs entry, or a public break without a
  Deviations entry.
