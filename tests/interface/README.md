# Interface tests

Tests for the interface described in `docs/design/interface.md`. They are written before the
interface exists and are independent of the rest of `tests/`.

Each test checks behaviour or arithmetic against a value worked out in the test itself (numpy,
or `jax.grad` of the equivalent pure function). There are no golden traces and no comparisons
with other frameworks.

## Reading a run

```bash
uv run pytest tests/interface -q --continue-on-collection-errors
```

- A test file that touches a missing part of the interface at import time (a trait it
  subclasses, a field it declares) fails at collection. The files are split by feature so that
  such a failure names one feature. `--continue-on-collection-errors` lets the other files run.
- Everything else is touched inside the test, so a missing feature fails only the tests that
  use it.
- These tests are collected only when a run names this directory or a file in it. A run of the
  rest of the suite (`pytest`, `pytest tests`) skips them, so the collection failures do not
  stop it. The hook that does this is in `conftest.py` and is to be removed once the interface
  is in place.

## Layout

| File | What it covers |
|---|---|
| `support.py` | building and running a small network, reading its state |
| `test_topology.py` | handle shapes, how many units and connections each layer makes |
| `test_forward.py` | forward arithmetic of dense, shared, convolution and sparse layers |
| `test_backward.py` | gradients and one optimizer step against `jax.grad` |
| `test_handle_notation.py` | `+ - *`, `sum`, `mean`, reshapes; softmax, layer norm, attention |
| `test_pooling.py` | max pooling forward and where its gradient goes |
| `test_policy_sets.py` | defaults, overrides on layers and handles, frozen sets |
| `test_fields.py` | namespaced and shared fields, reading another policy's field |
| `test_update_and_shared_weights.py` | contribution/combine/step, shared weights, per-weight state |
| `test_globals.py` | hyperparameters in global state, the step counter, `reset_global` |
| `test_rng.py` | `self.rng(g, element)` |
| `test_inputs_and_references.py` | named inputs, references to units |
| `test_accumulators.py` | struct accumulators, different accumulators in one network |
| `test_losses_and_outputs.py` | loss groups, outputs, custom losses |
| `test_initial_values.py` | layer initializers, policy-set `init` |
| `test_connection_pruning.py` | pruning, its place in the step, masking in dense layers |
| `test_connection_growth.py` | growth confined to a layer, network-level growth, capacity |
| `test_unit_lifecycle.py` | creating and removing units, `ReduceAll` |
| `test_running.py` | evaluation modes, propagation modes, formats |
| `test_build_errors.py` | every error raised when a network is built |
| `test_deepr.py` | the DeepR rule from `docs/design/sketches/deepr_multimnist.py` |
| `test_imprinting.py` | the feature unit and the TD loss from `sketches/imprinting_learner.py` |

## Undecided behaviour

A test of something the design has not settled is marked `open_question`, with the question as
the reason. Those tests are written against a best guess and are expected to change.

```bash
uv run pytest tests/interface -m open_question -q      # only those
uv run pytest tests/interface -m "not open_question"   # everything decided
```

## Interface the tests assume beyond the design document

Reading state back out of a network is not in the design document. The tests need it, so
`support.py` assumes the following and uses it in one place only:

- `px.UnitsRef(name)`: a reference to a group of units, bound with `handle.bind(ref)`, for
  reading state outside a step. (`px.UnitRef` is the single-unit reference policies use.)
- `px.inspect.unit_field(net, state, field, ref)`: a field's values over a group of units,
  with the shape of the handle. `ref` may be a `px.UnitsRef`, a `px.UnitRef` or a `px.Input`;
  with `None` it is every live unit, in unit order.
- `px.inspect.unit_ids(net, ref)`: the indices of those units.
- `px.inspect.connections(net, state, layer)`: the live connections of a layer (or of the whole
  network when `layer` is `None`), with `.src`, `.dst` and `.field(FIELD)`.
- `px.inspect.sources(net, layer)`, `px.inspect.destinations(net, layer)`: a layer's population.
- `px.inspect.num_units(net, state)`, `px.inspect.num_connections(net, state)`: live counts.
- `px.inspect.format(net, state, layer)`: the format a layer is stored in.

Also assumed:

- `@px.policy` makes a policy a dataclass: its attributes are constructor keywords
  (`Keeps(sign=-1.0)`), and a policy may subclass another.
- `px.init.constant`, `px.init.array`, `px.init.zeros`, `px.init.lecun_uniform`, and `bias_init=`
  on a layer.
- `px.SquaredErrorLossPolicy`: 0.5 * sum((y - t)^2), with seed y - t.
- `px.policies.Unit(init=...)` for initial unit field values.
- `px.BuildError` and `px.ShapeError` as the error types, and `px.PerformanceWarning`.
- `result.errors` as the non-fatal error queue of a step.
- `px.NO_LOSS` for a network with no loss.
- `px.UntieConnPolicy`, in a connection policy set as `untie=`.

## Not extracted

- SwiftTD's update. Checking it needs a reference implementation of the algorithm, which is a
  parity check.
- Training a whole sketch to an accuracy (the CNN, parallel MNIST, the transformer). These tests
  check single steps and short runs.
