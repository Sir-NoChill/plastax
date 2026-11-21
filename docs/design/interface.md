---
orphan: true
---

# Interface design

Status: draft for review. This document states the interface as agreed in discussion and in the
four sketches under `sketches/`, which are written against it and do not run.

Implemented so far (2026-11-21), all in `plastax.layers` and `plastax.init`: modules and tracing
(section 2.1), the two primitives and `Dense` with private weights (2.2), relabelling and
indexing of handles (2.3), weight initializers (2.8), and running a network that names its
topology, with one set of today's policies for the whole network (section 5). Nothing else
here is implemented.

- Names prefixed `px.` are proposals unless they exist in plastax today.
- A line beginning **Open:** is undecided. A line beginning **Proposed:** is a suggestion that
  has not been accepted. Both are collected in the last section.

## 1. Overview

A user specifies three things:

1. **Topology**: the initial structure.
2. **Behaviour**: policies for connections and for units, with their hyperparameters, constants
   and state fields.
3. **Hints and configuration**: predictions about the best representation, and plastax options.

`px.Network` is the union of the three and is the framework's interface.

<!-- fmt: off -->
```python
class Mnist(px.Network):
    topology = MnistCNN()                 # 1
    input_shape = (28, 28, 1)
    forward = SigmoidForward()            # 2: defaults, overridden on layers
    backward = SigmoidBackward(grad=GRAD, seed=SEED)
    optimizer = px.optim.adam(lr=1e-3, grad=GRAD)
    loss = px.policies.SoftmaxCrossEntropyLossPolicy(seed=SEED)
    config = px.Config(...)               # 3

driver = Mnist.init(key)
result = driver.step(px.StepInputs(inputs=image, targets=label))
```
<!-- fmt: on -->

**First principle.** The topology is the initial state, never a fixed structure. Units and
connections are created and destroyed at run time by policies. When a network is materialized,
its topology, policies and constants are flattened together, so any part of an algorithm can
refer to any other.

## 2. Topology

### 2.1 Modules and handles

A topology is a `px.layers.Module`, in the style of flax (not a dependency on it).

- `setup` declares layers. A layer is a set of connections.
- `__call__` receives handles to the input units, passes them through the layers, and returns
  the handles of the output units.

A handle (`px.layers.Units`) stands for a group of units and carries a shape. It carries no
data. Every operation on a handle happens while the network is being built.

```python
class MnistCNN(px.layers.Module):
    def setup(self) -> None:
        self.conv1 = px.layers.Conv(16, 3, padding="same", shared=True)
        self.conv2 = px.layers.Conv(16, 3, padding="same", shared=True)
        self.pool = px.layers.MaxPool(2)
        self.head = px.layers.Dense(10, hint=px.Hint(format=px.Format.DENSE))

    def __call__(self, x: px.layers.Units) -> px.layers.Units:
        h1 = relu(self.conv1(x))
        h2 = relu(self.conv2(h1) + h1)
        return self.head(self.pool(h2).flatten())
```

Every layer call and every handle operation is shape-checked at the line that wrote it.

Calling a topology computes nothing. `px.layers.trace(module, {"x": (28, 28, 1)})` creates the
input units, calls the module with one handle per input, matched to the arguments of
`__call__` by name, and returns a `px.layers.Record`: the blocks of units and of connections
the topology wrote down, as plain numpy. A record holds no weights and is the same every time
the topology is traced. `Network.init` (section 5) traces the topology and materializes the
record into a state.

- `setup` runs once, the first time the module is traced. A layer assigned to an attribute in
  `setup`, or held in a list or tuple assigned there, is registered under that name. Nested
  modules give a path, `("block1", "conv")` or `("blocks", "0", "attention")`, which names the
  layer in errors.
- A module takes its own arguments (`MnistCNN(width=16)`) in a plain `__init__` that only
  stores them.
- A layer is called once per trace. A second call raises `NotImplementedError` until shared
  weights give it a meaning.

Units are their index. There are no named populations, and a new unit is appended to the unit
arena if storage format allows.

### 2.2 Layers

A topology places units and connections. There are two primitives, one for each, and every
other layer is written with them.

<!-- fmt: off -->
```python
px.layers.CreateUnits(*, constant=None)      # called with a handle, or with shape=
px.layers.CreateConnections(*, init=None)    # called with (src, dst); returns nothing
```
<!-- fmt: on -->

- `CreateUnits` creates units and returns a handle to them. It creates no connections.
  `self.units(x)` creates as many units as `x` has, in its shape; `self.units(shape=(8,))`
  creates a group that stands on its own. `constant=1.0` makes units that always hold that
  value.
- `CreateConnections` connects units that exist. It creates no units. The two handles are
  broadcast against each other as arrays are, and each element of `src` is connected to the
  element of `dst` at the same place. Arranging the handles is how the wiring is chosen:

  <!-- fmt: off -->
  ```python
  self.skip(a, b)                                 # one to one, same shape
  self.full(a[None, :], b[:, None])               # every unit of a to every unit of b
  self.weight(x[..., None, :], y[..., :, None])   # the same along the last axis
  self.bias_weight(one, y)                        # one unit to every unit of y
  ```
  <!-- fmt: on -->

- Each call creates a **block**: a run of consecutive units, or a run of connections in
  row-major order over the broadcast shape. A block is named by the path of the primitive that
  made it, and it is what an initializer, a policy set, a capacity and a hint attach to.
- Two calls may connect the same pair of units, as two connections with two weights. The same
  pair twice within one call is an error.

Wiring that is not all-to-all is the same rule over handles indexed with integer arrays
(section 2.3). There is no `connectivity=` argument and no wiring-rule object:

<!-- fmt: off -->
```python
hidden = np.arange(N_HIDDEN)
self.readout(h[hidden], logits[hidden // GROUP])    # each hidden unit to one logit
```
<!-- fmt: on -->

`px.layers.Dense(features, *, bias=True, init=None, bias_init=None)` is a helper written with
the primitives. It acts on the last axis of its handle, once per position along the leading
axes, and every connection has its own weight. A bias is one unit that is always 1 and a
learned connection from it to each new unit, in a block of its own. A grid is flattened
explicitly before a dense layer that should read all of it: `self.head(p.flatten())`.

Further arguments carry structure and hints. None of these is built:

| Argument | Meaning |
|---|---|
| `shared=True` | positions use the same weights (section 3.8) |
| stride, padding | as in any array library |
| `conn_policy=`, `unit_policy=` | the policy sets for this layer, overriding the network's |
| `hint=` | a `px.Hint` (section 4) |

**Open:** which of these the primitives take, and which only the helpers.

**Open:** random initial wiring (`px.layers.sparse(density=...)`, `fan_in=...` in the
sketches). A topology is traced without a key. Either `trace` takes one, or a block holds a
description of its wiring that is resolved when the network is materialized, or random wiring
is numpy with a seeded generator and a helper that returns indices.

**Open:** where `capacity=px.Capacity.INITIAL` is written (section 3.10). It was an argument
of the wiring rules, which no longer exist. Most likely `CreateConnections`.

`px.layers.ReduceAll(n)` connects every feature unit to its `n` units, now and whenever a unit
is created. A connection of it that is pruned is never restored.

The network designer need not specify explicitly
that units may grow in a certain location in the
network. Plastax assumes that units and
connections are allowed to grow between arbitrary
units in the initial network structure. As units
or connections grow, it may erode some of the
structural qualities of the network which is
expected. The network may end up performing (in
terms of hardware-performance, not algorithmic
performance) worse by the end of learning.

### 2.3 Handle notation

The topology may be written with array notation on handles.

- `a + b`, `a - b`: new summing units fed by frozen connections with weights +1 and -1.
- `a * b`: one product unit per element (section 3.7).
- `h.sum(axis=...)`, `h.mean(axis=...)`, with `keepdims` and `scale`: summing units fed by
  frozen connections whose weight is 1, `1/n` or `scale`.
- `h[:, None, :]`, `h.reshape(...)`, `h.flatten()`: a relabelling of the handle's shape, in
  row-major order. They create nothing.
- `h[index]` with an integer array selects the listed units, as numpy does, and creates
  nothing. Integers for every axis, `h[0]`, select a single unit as a handle with no axes.
- Operands broadcast as arrays do.

The topology sets the weights of the connections these create. Their policy sets ship with
plastax and override the network defaults, as pooling's do.

`a + b` always means new summing units and returns a new handle; it never changes `a`. When the
network is built, a sum is merged into an operand that has no other reader and no activation,
so that operand's units compute the sum themselves. Where neither operand qualifies, the sum is
its own level of units. The goal being that when
plastax lowers through XLA, that these sum layers
might have their kernels fused with subsequent
layers if possible. Currently we are unsure if
this is possible under XLA for sparse matrix
multiplication, particularly in COO.

One line of this notation can create many units: `q[:, None, :] * k[None, :, :]` is one unit
per element of the broadcast shape.

### 2.4 Inputs

An input is declared once, as a module-level object, and that object is used wherever the input
is meant. The declaration is a named handle to a group of input units. An input unit is an
ordinary unit: it sits in the unit arena, has a policy set, and can be a source of connections
like any other. It differs in one way: its activation is written from the step's input data,
not by a forward policy.

<!-- fmt: off -->
```python
OBS = px.Input("obs")
GAMMA = px.Input("gamma")

class Learner(px.Network):
    inputs = {OBS: (num_obs,), GAMMA: ()}        # the network gives each a shape

step(state, px.StepInputs(inputs={OBS: frame, GAMMA: 0.99}))   # data is fed under it
```
<!-- fmt: on -->

The topology's `__call__` receives one handle per declared input, matched by name:
`px.Input("obs")` arrives as the argument `obs`. A network with one input may write
`input_shape = (28, 28, 1)` and take a single argument.

The type checker sees `px.layers.Units` for every argument. It does not know the names or the
shapes; the match is checked when the network is built. This is accepted as a limit of Python's
type system. (NOTE: to be revisited in a language with a more flexible type system allowing for
reflection and type elaboration at compile time)

### 2.5 References to units

A policy sometimes needs one particular unit (the unit carrying the discount, the readout).
Units have no identity at run time other than their index, so a reference is a name that exists
only while the network is being built.

- A reference is a module-level object, shared by the topology and the policies, in the same
  way a shared field is. Neither the topology nor the policies are parameterized on the other.
- A `px.Input` of a single unit is a reference to that unit.
- For a unit that is not an input, the topology binds a `px.UnitRef`:
  `self.readout(obs).bind(READOUT)`.
- A policy holds a reference as a typed attribute (`gamma: px.UnitRef`) and uses it wherever a
  unit index is expected: `u[px.ACTIVATION, self.gamma]`.
- When the network is materialized, every reference resolves to an index and the name is
  erased.

This is expected to be uncommon.

### 2.6 Policy sets on handles

Where a network has no global default, or a group of units differs from the global default, the topology gives
a handle its policy set:

```python
obs = obs.with_policy(observation)
```

`with_policy` returns a handle typed by the policy set. An activation applied to a handle
(`relu(h)`) is the common case of the same thing: it calls the network's unit policy with that
activation. A single policy can be overridden the same way: `identity(h, loss=ce)`.

### 2.7 Populations

Units have no population of their own. A population is defined by a layer: its sources and its
destinations, the unit indices the layer connected when it was built. A policy attached to a
layer (growth, for example) is handed a view of those two sets.

A unit created at run time can join a layer's population. Networks that do this (the later
`deep_r_progression` notebooks in `phd/structure_search`: growing outputs, inputs and hidden
units) are to be supported as first-class use cases. The mechanism is deferred and is
to be designed with the backend (card `unit-addition-into-layers`). The working position, by
format:

- **COO:** a connection finds its source and destination by index, so appending a unit to the
  unit arena costs more or less nothing. This is the form to go with for now.
- **CSR:** adding a unit is a rebuild, or at
  least a resort unless we can determine a smart
  way to pre-allocate unclaimed space in CSR.
- **Dense:** units cannot be added. The only alternative is rebuilding the dense layer
  completely, and then its shape changes. Shapes are static, so a rebuild also means recompiling
  the step function.

### 2.8 Initial values

The topology sets a connection's initial weight. A block of connections has an initializer
(`px.layers.Dense(300, init=px.init.he_normal())`). The record stores it, and it is called
once per block when the network is materialized.

- An initializer is any function `(key, fans) -> weights`. `px.init.Fans` carries the block's
  shape and, for every connection of the block, the fan-in of its destination and the fan-out
  of its source. The weights returned have the block's shape.
- The fans are counted from the wiring, over the whole record: the fan-in of a connection is
  the number of connections into its destination from every block. A layer with 4 inputs per
  unit is therefore drawn with a fan-in of 4, whatever its shape. The fan-in leaves out
  connections from constant units, so a bias does not count; the fan-out counts every
  connection. A fan of 0 is read as 1.
- Every shipped initializer is called where it is written: `px.init.zeros()`,
  `px.init.lecun_uniform()`, `px.init.constant(0.25)`, `px.init.array(W)` with `W` in the
  block's shape. Shipped: `constant`, `array`, `zeros`, `ones`, `normal`, `truncated_normal`,
  `uniform(low, high)`, `variance_scaling(scale, mode, distribution)` and the LeCun, Glorot
  and He initializers written with it. `orthogonal`, `delta_orthogonal` and `from_jax` are
  declared and raise `NotImplementedError`.
- A block with no initializer is drawn with `px.init.lecun_normal()`. A `Dense` bias starts
  at `px.init.zeros()`.
- Each block's key is derived from the key given to `init` and the block's path. The same key
  and the same topology give the same weights, and adding a layer does not change the weights
  of the others.

A connection policy set's `init` fills in the other fields, either as fixed values or as a
function. The function is handed the globals and the drawn weight, for a field that depends on
a hyperparameter or on the weight. It is traced code.

A layer that creates connections at run time (`ReduceAll`) uses its initializer for those too.
A connection created by a growth policy has no layer initializer; its values come from that
policy's `init`.

**Open:** how an initializer is called for connections created at run time, inside the
compiled step, and what the fan of a shared weight is.

A policy set never sets the weight. Frozen identity connections get their weight from whatever
creates them: the `MaxPool` layer and the handle operations in the topology, the growth
policy's `init` for imprint connections.

We should note that we will provide standard
initializers and default initialization within
plastax, similar to the initialization practices
provided by pytorch and JAX.

## 3. Behaviour

### 3.1 Phases, pairs and defaults

Behaviour is a grid of phase by element: for each phase, a policy written for one connection
and a policy written for one unit. A network declares it by phase. Each phase is a pair with a
`conn` half and a `unit` half (owner, 2026-11-21; this replaces the policy sets
`px.policies.Conn` and `px.policies.Unit`, which grouped the same grid by element).

<!-- fmt: off -->
```python
class Net(px.Network[Globals]):
    forward = px.policies.Forward(conn=WeightedSum(), unit=SigmoidForward())
    backward = px.policies.Backward(conn=WeightedSumBackward(), unit=SigmoidBackward())
    loss = MSELoss()
    optimizer = px.optim.adam(lr=1e-3, grad=GRAD)          # its update and its state fields
    update = px.policies.Update(unit=RollLagged())         # the half the optimizer leaves
    prune = px.policies.Prune(conn=OlderThan(AGE, 1000), unit=OrphanPrune())
    grow = px.policies.Grow(conn=RandomProposals(), unit=SplitSaturated())
    reset_global = CountSteps()
```
<!-- fmt: on -->

- `Forward` and `Backward` require both halves. `Update`, `Prune` and `Grow` take either half
  or both, and one with neither is refused. Every field but `forward` may be `None`, which
  elides the phase.
- One object that has `map`, `combine` and `apply` may be given as `forward` or `backward`.
  When the class is defined the field is replaced by the pair with that object in both halves,
  so `Net.forward.conn` and `Net.forward.unit` always exist.
- An optimizer is a bundle: `conn_fields` and `unit_fields`, the state only it reads, and
  `update()`, an `Update` pair. The network adds the fields to its columns and takes the
  update. A half the optimizer supplies cannot also be set in `update`: a network has one
  connection update and one unit update. The network's own state is listed in
  `extra_unit_fields` and `extra_conn_fields` until typed fields (section 3.4) replace them.
- What a network declares is the default. A layer is given policies the same way a network
  is, inherits the network's, and may override them. This is not built, and its sections
  (2.2, 2.6, 3.6, 3.9) are still written for policy sets. Open: a layer's `backward.conn` is
  reduced at the previous layer's units, so the two layers need a contract on the
  accumulator, ideally one that types can check.

Dispatch between the policies present in one sweep is a switch: every case is computed for
every element and one is selected. Laying elements out by policy is a later optimization.

Packages: `plastax/policies/` (every policy trait, the pairs and the shipped policies; none is
exported at the top level), `plastax/activations/`, `plastax/layers/`. Every trait name ends
in `Policy`.

### 3.2 Policy methods

Every policy method is written for one element (one unit or one connection) and the framework
maps it over all of them. Every index and value is therefore a scalar. The loss is the
exception: it sees a group of units at once.

| Argument | Type | Meaning |
|---|---|---|
| `u` | `px.UnitView` | read-only view of all unit state: `u[FIELD, unit_index]` |
| `c` | `px.ConnView` | read-only view of all connection state: `c[FIELD, conn_index]` |
| `i`, `src`, `dst`, `parent`, `child` | `px.UnitIdx` | one unit's index |
| `cid` | `px.ConnIdx` | one connection's index |
| `g` | the network's globals | global state, read-only |

A policy never assigns into `u` or `c`. It returns a write (`px.UnitWrite`, `px.ConnWrite`): a
set of (field, new value) pairs for the one element it was called on. Fields not named keep
their value.

We should note that there exists a mechanism to
modify global state, however that mechanism is
likely to severely decrease performance. For that
reason, global state is normally offered on a
read-only basis, so that performance remains good
across all networks that fall within the plastax
model (all behaviour on connection/unit local
scopes, rather than on layer scope).

### 3.3 The traits

The forward and backward passes are each split into a per-connection half and a per-unit half.

<!-- fmt: off -->
```python
class ForwardConnPolicy[Acc, G](Protocol):
    def map(self, u, dst, src, c, cid, g) -> Acc: ...        # one connection's contribution

class ForwardUnitPolicy[Acc, G](Protocol):
    combine: Monoid                                          # how contributions are reduced
    def apply(self, u, i, g, acc: Acc) -> px.UnitWrite: ...

class BackwardConnPolicy[Acc, G](Protocol):
    def map(self, u, src, dst, c, cid, g) -> Acc: ...
class BackwardUnitPolicy[Acc, G](Protocol):
    combine: Monoid
    def apply(self, u, i, g, acc: Acc) -> px.UnitWrite: ...
```
<!-- fmt: on -->

The connection update is split for every weight into a contribution per use, a combine, and one
step. A private weight is the case of one use.

<!-- fmt: off -->
```python
class UpdateConnPolicy[Total, G](Protocol):
    combine: Monoid = sum_
    def contribution(self, u, dst, src, c, cid, g) -> Total: ...   # per use of the weight
    def step(self, c, cid, g, total: Total) -> px.ConnWrite: ...   # once per weight
```
<!-- fmt: on -->

`step` returns a write and not a new weight, because an update may change many fields of its
connection.

Fields declared by an `UpdateConnPolicy` belong to the weight. They are stored with it, whether
it is private or shared, and `step` reads them through `c` exactly as it reads `px.WEIGHT`. An
optimizer's moments are therefore one value per weight, not one per use.

The remaining traits keep today's shape with the `Policy` suffix: `UpdateUnitPolicy.update`,
`PruneConnPolicy.predicate`, `PruneUnitPolicy.predicate`, `AddUnitPolicy.spawn` and `.init`,
`ScoreAddConnPolicy` and `ProposeAddConnPolicy` with their `init`, `LossPolicy.calculate_loss`.

One step runs the phases in this order: forward, loss, backward, update units, update
connections, prune units, prune connections, add units, add connections.

Plastax does not differentiate policies automatically. The backward pass is the policies a user
writes.

### 3.4 Fields, constants and hyperparameters

A field is a column of unit or connection state, declared with its type.

- A field declared inside a policy is namespaced by that policy:
  `pre: px.UnitField[np.float32] = 0.0`. Two policies may each have a `pre`.
- A field shared between policies is one module-level object, used by each:
  `GRAD = px.UnitField[np.float32]("grad_pre_act", default=0.0)`.
- The framework collects the fields from the policies in use.

Policies depend on each other. The dependency is written as a typed attribute and wired by
hand, so the type checker sees it.

- Pass the **field** when only the other policy's state is needed:
  `IdleFeaturePrune(dropped=update.dropped)`.
- Pass the **policy** when its behaviour is needed:
  `ActivationBackward(forward=forward)` calls `forward.activation.derivative`.

Hyperparameters:

- Numeric hyperparameters live in global state and are read by policies as `g.hp`. They are
  run-time values, so changing one does not recompile.
- Values that fix a shape or are used in plain Python (table lengths, capacities, shortlist
  sizes, which side of a product a connection fills) are static: module constants or plain
  policy attributes. Changing one recompiles the step.
- There is no `px.Hyper` type. A value is one or the other of these two.

### 3.5 Global state and randomness

Global state is read-only. No policy writes it, and a value that changes every step is not held
there. A per-step value is an input, carried by a unit. The framework supplies `g.step` and
`g.seed`.

The exception is `reset_global`, kept for an algorithm that cannot do without writable global
state. When set it runs as the last phase of every step and returns a new global state. Using
it raises a warning when the network is built, because it significantly reduces performance.

Every policy can draw random numbers. The stream is split per policy: a policy's `rng` is its
own stream for this element and this step, independent of every other policy's, and draws
replay exactly. Two methods of the same policy called for the same element in the same step
get the same stream from the same starting point.

A policy that needs randomness asks for it:

<!-- fmt: off -->
```python
def step(self, c, cid, g, total) -> px.ConnWrite:
    rng = self.rng(g, cid)          # this policy's stream, for this connection and this step
    noise = rng.normal()
```
<!-- fmt: on -->

- `self.rng(g, element)` is a method every policy has. `element` is the index the method was
  called for: a unit, a connection, or a proposal number.
- No trait signature carries an `rng`, so only the policies that use randomness mention it,
  and the call is fully typed.
- The framework gives each policy its stream when the network is built. The stream is derived
  from `g.seed`, `g.step`, the policy and the element.

### 3.6 Accumulators

A connection policy's `map` produces a value; the unit policy at the other end combines those
values and receives the result in `apply`. The two must agree on the accumulator. A connection
has two: its forward accumulator must match its destination's unit policy, and its backward
accumulator its source's. Each trait states its accumulator as a type parameter.

A sweep reduces every connection into one accumulator. When policy sets have different
accumulators, the network's accumulator is their product: a struct with one part per distinct
accumulator, which is itself a monoid. The framework builds it. A connection fills its own part
and leaves the identity in the others; a unit's `apply` is handed its own part.

A connection attached to a unit that reads a different part would be silently ignored. That is
never permitted: a mismatch is an error, raised when the network is built. Connections created
at run time are covered by the same error, because a growth rule states what it can create:

- Growth carried by a connection policy set (`add=`) creates connections of that set, from the
  layer's sources to the layer's destinations.
- Network-level growth names the connection policy set it creates and the unit policy set it
  may connect into: `ImprintWiring(creates=imprint, into=feature)`.

A connection half may declare `linear_input = F`: its `map` is exactly the weight times the
unit field `F` at the other end. Paired with a unit half whose `combine` is the additive
monoid of a semiring the engine has a sparse product for, the pass is computed as one sparse
matrix product. A semiring is `px.monoid.LinearSemiring(add=, mul=)`, named by its additive
and multiplicative monoids; the engine's are `px.monoid.LINEAR_SEMIRINGS`, today
`plus_times` (`add=sum_`, `mul=prod`) only. Any other pairing warns when the class is
defined and runs per connection.

Static type checking of the pairing may be added later. It would need the accumulators to be
distinct classes, and it cannot see connections created at run time.

### 3.7 Activations, pooling and product units

An activation is a function and its derivative.

<!-- fmt: off -->
```python
class Activation(Protocol):
    def __call__(self, h, **unit_overrides): ...   # on a handle: sets the units' activation
    def fn(self, x): ...
    def derivative(self, x, y): ...                # from pre-activation x and output y
```
<!-- fmt: on -->

Each shipped activation has an analytical derivative; a hand-written one may fall back to
`jax.grad`. The derivative need not be the true one (LTU: a step forward, a sigmoid derivative
backward).

**Max pooling** ships its own policy sets as a layer-wide override. The forward reduce is a
plain max and no winner is stored. Each pooling connection's backward `map` passes the gradient
when its source's activation equals its destination's, so tied inputs each receive the full
gradient.

**A product unit** has exactly two incoming connections, a left factor and a right factor, and
outputs their product. Its accumulator is a pair, each side combined by the `prod` monoid; a
connection fills its side and puts 1 in the other. The unit keeps both factors, so each
connection's backward contribution is the gradient times the other factor. Attention, softmax
and layer normalisation need nothing else: they are sums, products and the activations `exp`,
`reciprocal` and `rsqrt`.

A layer of this kind that should run as tensor operations gets a layer builder
(`px.layers.Attention`) that creates the units and knows how to run them. The framework does
not try to recognise the pattern. A transformer is a distant use case.

### 3.8 Shared weights

A shared weight is presented as a shared weight connection. `px.layers.Conv(..., shared=True)`
creates one shared weight per kernel entry and one connection row per use, each referencing the
weight. `c[px.WEIGHT, cid]` reads the weight whether it is private or shared.

- Shared weights are local state, stored apart from globals.
    - Well, in the implementation this will be
      something like storing global state anyways,
      unless there is a clever way we could use
      some ephemeral data structure to extract it
      from a local unit into the current execution
      context... but for now we can assume that
      this is akin, in implementation, to a global
      state, but that we want to think of it as
      still being local, just that the connection
      has an enforced equality between all edges.
- The update sums the per-use contributions and takes one step (section 3.3).
- In a sparse format, pruning one use prunes that connection, and untying copies the value onto
  it as a private weight.
- Untying is one-way. A use that has been untied stays untied; there is no retying.

### 3.9 Frozen connections

There is no frozen flag. A connection keeps the policy set it was created with, and a frozen
connection is one whose set has no update, prune or growth policy. A frozen set says so
explicitly, overriding the network's defaults with `None`:

```python
pool_conn = px.policies.Conn(
    forward=WeightedSum(), backward=MaxPoolBackward(), update=None, prune=None, add=None
)
```

An algorithm that needs to freeze and unfreeze implements that within its own policy.

### 3.10 Growth, pruning and capacity

Growth has two forms, following the rule for every policy: the network's is the default and a
layer's overrides it.

- The network's `add_conn` may connect any two units. It names what it creates and what it may
  connect into (section 3.6).
- A connection policy set may carry its own growth (`add=`). It applies to the layer the set is
  attached to: a new connection goes from that layer's sources to that layer's destinations,
  belongs to that set, and uses that layer's slots.

A compiled step cannot raise. A network-level growth proposal into a unit outside the rule's
`into` set is not created, and is emitted as a non-fatal error: it is added to an error queue
that the step returns. **Open:** how the queue is held on the device and read back.

A policy that creates units names the policy set of its children statically
(`ImprintSpawn(child=feature)`). A grown unit does not inherit from its parent.

A layer that must keep its connection count has exactly as many connection slots as it starts
with connections. Pruning frees a slot and growth fills free slots, so no counter is held in
global state. The count is exact whenever a step offers at least as many usable proposals as it
freed slots; a shortfall is refilled on a later step.

Structure changes by format:

| Format | Connections | Units |
|---|---|---|
| COO | appended | appended |
| CSR | rebuilt | rebuilt |
| DENSE | masking | never change |

In a dense layer a pruned connection is masked out and a connection grown back inside the
layer's block is unmasked. A connection grown between two units that the dense block does not
cover is stored in a sparse format beside it. Units are never added to a dense block.

**Open:** masking one use of a shared weight in a dense convolution. A native convolution
cannot leave out a single use, so this needs either the correction connections of card
`dense-tap-exceptions` or falling back to a sparse format for that layer.

### 3.11 Losses and outputs

- `outputs` of a step has the shape the topology's `__call__` returned.
- A loss sees a group of units. For outputs that are a regular grid of groups, the groups are
  given by the handle's shape: the logits are reshaped to `(task, class)` and one loss runs
  over the last axis (`px.SoftmaxCrossEntropyLossPolicy(axis=-1)`).
- For irregular outputs a loss is applied to a handle (`identity(h, loss=ce)`); the group is
  that handle's units, and when the topology returns several handles, targets are matched to
  them by position. Accepted for now, to be improved if a better form is found.

The reshape that defines the groups is checked: where it is written (the new shape holds
exactly the handle's units), when the network is built (the loss's axis exists), and at a step
(the targets have the shape of the outputs). What cannot be checked is that row-major order
puts the intended units in each group; that depends on how the layer before was wired.

## 4. Hints and configuration

<!-- fmt: off -->
```python
px.Hint(format=px.Format.DENSE)                 # on a layer
px.Config(format=px.Format.AUTO,                # on the network: the defaults
          propagation=px.Propagation.TOPOLOGICAL,
          representation_optimization=False,
          unit_capacity=4096, seed=0)
```
<!-- fmt: on -->

- `px.Format` is `AUTO`, `COO`, `CSR` or `DENSE`. A format on a layer is a prediction by the
  user of what will run best; the network's is the default.
- Representation optimization, the framework changing formats on its own, is opt-in and off by
  default.
- `unit_capacity` is how many units the arena can hold. Capacity for growth is configuration.
- `propagation` is the topologically sorted mode or the pipeline mode, in which every unit
  reads the previous step's values and there is no ordering to maintain.
- The topologically sorted mode is built first, because it is the one most likely to cause
  problems; the pipeline mode is easy and follows. The implementation takes its cues from
  the design that exists today and modifies it to reach this interface.
- Options that are strings today (`layout`, `fuse_prune`, `growth`, growth selection and
  candidates) become enums.

Static metadata: shapes, capacities and level counts are Python ints, because the compiled step
is keyed on them. Data that is a tuple today, such as input and output ids, can be arrays.

Build-time code uses numpy. `jax.numpy` is used only for code that runs in, or is shared with,
the compiled step.

## 5. Running a network

<!-- fmt: off -->
```python
class Learner(px.Network[Globals]):
    topology = Imprinting()
    input_shape = (num_obs,)
    globals = Globals(hp=Params())            # the value the global state starts from
    ...

driver = Learner.init(jax.random.key(0))
result = driver.step(px.StepInputs(inputs=frame, targets=reward))     # result.loss
result = driver.scan(px.StepInputs(inputs=frames, targets=rewards))   # result.loss: (K,)
```
<!-- fmt: on -->

- `init` materializes the network: the topology becomes units and connections, references
  resolve, and the build-time checks run. It returns a `px.Driver`.
- The driver owns the network's state and its static configuration, and the iteration is
  the framework's. A network that changes its structure outgrows its storage and has to be
  resorted; both replace the static configuration and recompile the step. The driver does
  this, so the user's loop never sees it.
- `driver.step` runs one step. The loop around it is the user's. Each call costs a dispatch
  and a read of two flags back from the device.
- `driver.scan` runs a chunk of `K` steps in one loop on the device, exactly as `K` calls of
  `step` would, and comes back once per chunk. It is the way to run a small network. The
  chunk is given along a leading axis.
- Both return a `px.RunResult` with `loss`, `outputs` and `records`. From `step` each is one
  step's value; from `scan` each has a leading axis of length `K`.

**Inputs and targets** are given in their declared shapes: `inputs` in `input_shape`,
`targets` in the shape the topology returned. The step flattens them in row-major order.
Both travel in a `px.StepInputs`. A wrong shape is an error naming the declared one.

**Open:** `StepInputs` as an alias for whatever the network requires, in the manner of an
associated type. With named inputs (section 2.4) a step's inputs are a mapping, so their
type depends on the network.

**Outputs are returned only when asked for**, with `outputs=True` on `step` or `scan`. They
are read right after the forward phase: what the forward policies computed for this input,
before the loss, the backward pass, any update and any structural change. What an output
unit holds at the end of a step is not that in general, in a network that modifies itself
least of all. Not asked for, the step contains no trace of the read. A batched step cannot
return them yet.

`scan(..., record=fn)` keeps any other per-step signal: `fn` is evaluated on the device for
every step and its values come back as `result.records`. The same function object has to be
passed every time, since the compiled loop is kept per function.

**Globals.** The value the global state starts from is a class field, `globals`. It is made
of Python numbers or numpy arrays, and it is never mutated, since every network of the
class starts from it; `init` copies it onto the device. `Learner.init(key, globals=...)`
replaces it for one network, for a sweep. `reset_global` (section 3.5) rewrites the value
held in the state and never the class field.

**The functional layer** stays public, for whoever needs `vmap` or an outer `jit`:

<!-- fmt: off -->
```python
static, state = Learner.build(jax.random.key(0))
step = px.make_step(Learner, static, outputs=True)
result = step(state, px.StepInputs(inputs=frame, targets=reward))   # a px.StepResult
```
<!-- fmt: on -->

- `build` is what `init` calls; `init` is `px.Driver(Learner, *Learner.build(key))`.
- The step consumes the state it is given: the state is donated, and the old value cannot
  be read afterwards.
- The caller reads `result.overflow` and `state.needs_resort` and rebuilds the step after
  a regrow or a resort, with `grow_bucket`, `topo.resort` and `make_growth_retry`. A flag
  that is ignored raises nothing: growth candidates are dropped, and in the sorted mode a
  connection that breaks the level order may be skipped by the forward pass.

Limits of what is built (2026-11-21):

- One set of today's policies applies to every unit and every connection.
- One input (`input_shape`) and one returned handle. Named inputs and several outputs are
  not built.
- The topologically sorted mode only: a constant unit keeps its value because a unit that
  is not an input and has no incoming connection is never written there.
- A cycle, or a connection into a constant unit, is a `BuildError` naming the block.
- The engine options (`batch_size`, `layout`, `fuse_prune`, `growth`, `check_every`) are
  arguments of `px.Driver`'s constructor and `init` passes none. They move to `px.Config`.

`px.layers.trace` and `px.layers.materialize` remain public, for a record traced by hand.

Today's `NetworkBuilder`, `Block` and `sequential` are removed outright. Modules and layers
replace them and are built anew.

Evaluation is the forward policies alone. Forward policies can write state, so there are two
modes:

```python
predict = px.make_step(Mnist, phases=px.Phases.FORWARD, writes=px.Writes.DISCARD)
```

- `px.Writes.DISCARD`: run forward, return the outputs, discard what was written.
- `px.Writes.KEEP`: run forward and keep what it wrote, for networks whose state advances on
  every input.

## 6. Errors raised when the network is built

These are raised before the first step, naming what the user wrote. A shape mismatch at a layer
call, a handle operation, a `reshape` or a `flatten` is raised earlier, at the line that wrote
it.

- A network policy set to `None` without every layer or handle supplying its own.
- An input declared on the network with no matching `__call__` argument, or the reverse.
- A reference that is never bound, bound more than once, or bound to more than one unit.
- A referenced unit whose policy set can prune it.
- A connection attached to a unit whose policy has a different accumulator.
- A growth rule whose connection policy set and destination unit policy set have different
  accumulators.
- A loss whose axis does not exist in the shape the topology returns.
- A policy that adds units to a layer hinted `DENSE`, or that unties a use of a shared weight
  in one. Pruning and growing connections there is allowed (section 3.10).
- A hint the layer cannot honour.
- An activation or policy that cannot be compiled.
- A field declared with conflicting types.
- Growth policies present with no unit or connection capacity to grow into.

A warning, not an error: `reset_global` is set.

**Open:** the error types, and what each message names.

## 7. Deferred

- **Batching** comes last. Today's batched step and batch reductions are not part of this
  interface yet.
- **Sharding** is left as it is and allowed to break.
- **Unit addition into existing layers**: the mechanism (section 2.7).
- **Weights as references.** A connection's weight could be a reference with three targets: its
  own slot, a shared weight, or a unit's activation. The third is a connection that multiplies
  two activations with half the units a product unit needs, in the shape of the shared-weight
  update. It costs a second reduce in the backward pass, and in the sorted mode the referenced
  unit may be deeper than the connection's source, so its gradient would arrive too late. Not
  adopted.
- **The imprinting learner's readout.** The running example has one; whether every prediction
  should be a general value function is to be tested once the implementation runs.

## 8. Open and proposed

1. **How refused growth proposals are queued and read** (section 3.10).
1. **Masking a use of a shared weight in a dense convolution** (section 3.10).
1. **Whether XLA fuses a sum's level of units with the layer after it** in the sparse formats
   (section 2.3). To be measured in the feasibility benchmark.
2. **The error types, and what each message names** (section 6).
3. **Deferred items** (section 7).
4. **Which arguments the primitives take** (`conn_policy=`, `unit_policy=`, `hint=`,
   `shared=`), and which only the helpers (section 2.2).
5. **Random initial wiring**, and where its key comes from (section 2.2).
6. **Where a block's capacity is written** (section 2.2).
7. **Initializers for connections created at run time, and the fan of a shared weight**
   (section 2.8).
