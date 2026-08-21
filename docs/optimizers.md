# Optimizers as trait bundles

In plastax an optimizer is not a special object living outside the network --
it is a *trait bundle* wired in like any other policy. A bundle (the
`plastax.optim.Optimizer` protocol) is three things:

- an `UpdateConn` policy that computes the weight step (built by
  `optimizer.update_conn()`),
- the extra per-connection state columns that step needs (`state_fields`), and
- whether it needs a globals step counter (`needs_step_counter`).

Because the optimizer's state lives as per-connection SoA columns -- Adam's
first and second moments, momentum's velocity -- it shards with the connections
under Scheme-A sharding with no separate optimizer-state plumbing: **a
distributed optimizer for free**.

## The gradient

Every optimizer forms the per-edge weight gradient by the delta rule,
`dL/dw = grad_pre_act[dst] * activation[src]`, which is exact for any
weighted-sum layer (a dense layer or the unrolled convolution of
`plastax.topology.conv2d`). You tell the optimizer which unit field your
backward pass writes `dL/dz` into via its `grad_field` argument.

## Wiring one in

An optimizer contributes its `state_fields` to the network's
`extra_conn_fields`; everything else is a normal trait assignment:

```python
import plastax as px

# the unit field your backward pass writes dL/dz into (see examples/mlp_xor.py)
Delta = px.FieldSpec.float32("grad_pre_act")
opt = px.optim.adam(1e-3, Delta)

class Net(px.Network[None]):
    forward_pass = MyForward()
    backward_pass = MyBackward()
    loss = MyLoss()
    update_conn = opt.update_conn()
    extra_conn_fields = opt.state_fields  # () for sgd; (opt/m, opt/v, opt/t) for adam
    propagation = px.Propagation.TOPOLOGICAL
```

## Available optimizers

`sgd`, `momentum`, `adam`, `adamw`, and `rmsprop`. Each is validated against its
optax reference to float32 precision, online and on MNIST (`tests/test_optim.py`
and `examples/mnist_sgd.py`). optax is used only as a test oracle -- it is never
a runtime dependency.

## Reference

```{eval-rst}
.. automodule:: plastax.optim
   :members:
   :imported-members:
```
