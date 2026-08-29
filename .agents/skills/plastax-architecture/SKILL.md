---
name: plastax-architecture
description: >-
  Give an agent the architectural layout of the plastax codebase and route a
  contribution to the module that should own it. Use this whenever you are new
  to the repo, unsure where a change belongs, or asking "where do I add X",
  "which file owns Y", "how does the step function get assembled", "what depends
  on what", or "what's the difference between NetworkStatic and NetworkState".
  Use it before starting any non-trivial change so the change lands in the right
  layer and respects the dependency stack. Prefer this over guessing from file
  names. Pairs with plastax-algorithm-scaffold (for the how-to of a specific
  contribution) and plastax-review (before committing).
---

# plastax architecture overview

Orient in the plastax codebase and route a change to the right module. plastax
expresses highly sparse, dynamically structured networks as declarative traits
over a struct-of-arrays edge arena, assembling a jitted step function at trace
time. This skill points you at the authoritative layout and helps you place a
change correctly.

## Step 1 — Load the map

Read, in this order:

1. [`AGENTS.md`](../../../AGENTS.md) — the binding rules, the module map table,
   the golden invariants, and the commit/hook workflow. Everything starts here.
2. [`.agents/architecture.md`](../../architecture.md) — the full
   architecture: the dependency stack, the two-tier state model, the sweep
   engine, the trait → phase → step assembly path, the **contribution routing
   table (§6)**, the optimizer bundle contract (§7), and testing/example
   conventions (§8–9).
3. [`.agents/invariants.md`](../../invariants.md) — the eight
   non-negotiables. A change that appears to require breaking one is almost
   always misrouted.
4. [`.agents/glossary.md`](../../glossary.md) — any unfamiliar term
   (arena, tombstone, monoid, bucket, phase elision, null-slot trick, retrace,
   churn step, Scheme-A/B, delta rule).

Do not re-derive the layout from scratch by reading `src/` — these docs are the
distilled map and are kept in sync with the code. Drop into the cited
`file.py:line` only to confirm a specific signature.

## Step 2 — Route the change

Answer three questions:

1. **Is this a new *algorithm* (a learning rule, loss, prune/grow policy,
   optimizer, topology) or a change to the *framework*?**
   - New algorithm ⇒ it is almost always a **new policy class implementing an
     existing Protocol** (or an optimizer/topology addition) and touches **no
     framework module**. Switch to the **`plastax-algorithm-scaffold`** skill.
   - Framework change ⇒ continue.
2. **Which layer of the dependency stack does it belong to?** Use the routing
   table in `architecture.md` §6 and the module map in `AGENTS.md`. Respect the
   stack: lower layers (`_types`, `monoid`, `state`, `views`) never import upper
   ones. A change that would add an upward import is misrouted.
3. **Which invariant governs it?** Name the invariant(s) from `invariants.md`
   your change must preserve (shape-static / donation / phase-elision /
   null-slot / derived-counts / pure-policies / mypy-strict / v1-scope) before
   writing code.

## Step 3 — State the plan before coding

Produce a short routing decision the user can sanction:

- The module(s) you will touch and why that module owns this.
- The public-API impact (does it touch `src/plastax/__init__.py`'s `__all__`? —
  if so it is a `type(scope)!:` change with a Deviations entry).
- The commit **scope** (from `SCOPES.md`) — one scope per commit; if the change
  naturally spans scopes, that is a signal to split it.
- The invariants in play and how you keep them.
- The tests and docs the change requires (docs-in-sync is a review condition).

Then implement, and run the **`plastax-review`** skill before committing.

## Quick reference — the module map

| Layer | Modules | Owns |
|---|---|---|
| Host loop | `driver.py` | retrace / overflow / resort control flow |
| Assembly | `step.py`, `builder.py` | jit/donate/shard-map; eager construction |
| Compiler | `phases.py` | traits → ordered pure phases (elision) |
| Engine | `sweep.py`, `topo.py` | gather/map/reduce/apply; levels/resort/capacity |
| Declaration | `traits.py` | `Network` base + policy Protocols |
| Substrate | `state.py`, `views.py`, `monoid.py` | arena state; views/writes; combine algebra |
| Leaf | `_types.py` | NewTypes, `FieldSpec`, enums |
| Generators | `topology.py`, `shard.py` | host-side topology DSL; Scheme-B partition math |
| Optimizers | `optim/` | optimizer bundles (`sgd`…`rmsprop`) |

The overwhelmingly common contribution — a new plasticity algorithm — is a new
policy class and touches none of these. When in doubt, route to
`plastax-algorithm-scaffold`.
