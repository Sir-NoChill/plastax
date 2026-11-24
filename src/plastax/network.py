"""The Network base class and its class-definition checks."""

from __future__ import annotations

import dataclasses
import inspect
import warnings
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PRNGKeyArray

from plastax._types import (
    ACTIVATION,
    DEAD,
    FROM_ID,
    LEVEL,
    PRUNED,
    TO_ID,
    WEIGHT,
    FieldSpec,
    Propagation,
    ShardSpec,
)
from plastax.monoid import LINEAR_SEMIRINGS, Monoid
from plastax.optim import Optimizer
from plastax.policies import (
    INHERIT,
    AddUnitPolicy,
    Backward,
    BackwardConnPolicy,
    BackwardUnitPolicy,
    BatchReductionPolicy,
    FieldReductions,
    Forward,
    ForwardConnPolicy,
    ForwardUnitPolicy,
    Grow,
    Inherit,
    LossPolicy,
    ProposeAddConnPolicy,
    Prune,
    PruneConnPolicy,
    PruneUnitPolicy,
    Reduction,
    ResetGlobalPolicy,
    ScoreAddConnPolicy,
    Update,
    UpdateConnPolicy,
    UpdateUnitPolicy,
    batch_written_reduction,
)
from plastax.policies._batch import _STRUCTURAL_UNIT_FIELDS

if TYPE_CHECKING:
    from plastax.driver import Driver
    from plastax.layers import Fault, Module, PolicyTable, Record
    from plastax.state import NetworkState, NetworkStatic


@dataclasses.dataclass(frozen=True)
class Resolved[GS]:
    """A network's optional policies and fields, one attribute per cell.

    Type Args:
        GS: the global state type threaded through the network.

    Attributes:
        update_conn: the connection update, from `update` or the optimizer.
        update_unit: the unit update, from `update` or the optimizer.
        prune_conn: the connection pruning policy.
        prune_unit: the unit pruning policy.
        grow_conn: the connection growth policy.
        grow_unit: the unit addition policy.
        unit_fields: the unit fields beyond the builtin ones: the network's
            `extra_unit_fields`, then the optimizer's.
        conn_fields: the connection fields beyond the builtin ones, likewise.
    """

    update_conn: UpdateConnPolicy[GS] | None
    update_unit: UpdateUnitPolicy[GS] | None
    prune_conn: PruneConnPolicy[GS] | None
    prune_unit: PruneUnitPolicy[GS] | None
    grow_conn: ScoreAddConnPolicy[GS] | ProposeAddConnPolicy[GS] | None
    grow_unit: AddUnitPolicy[GS] | None
    unit_fields: tuple[FieldSpec[np.generic], ...]
    conn_fields: tuple[FieldSpec[np.generic], ...]


class Network[GS]:
    """Base configuration surface for a network's traits.

    Subclass and set class attributes; absent phases are elided at trace
    time.

    Type Args:
        GS: the global state type threaded through the network.

    A subclass may name its initial structure as `topology`, with `input_shape`;
    `build` then gives the `(static, state)` pair and `init` a `Driver` that runs
    it. A subclass without a topology is built with `NetworkBuilder`.

    Attributes:
        topology: the network's initial structure, a `plastax.layers.Module`
            taking one handle, or None for a network built with
            `NetworkBuilder`.
        input_shape: the shape of the topology's input units. A step's inputs
            have this shape.
        globals: the value the global state starts from, or None. A pytree of
            Python numbers or numpy arrays. Every network built from the class
            starts from it, so it must not be mutated; `build` copies it onto
            the device.
        forward: the forward pass, a `plastax.policies.Forward` pair of a
            connection half and a unit half. One object that has `map`,
            `combine` and `apply` may be given instead; when the class is
            defined it is replaced by the pair with that object in both
            halves, so `forward.conn` and `forward.unit` always exist.
        backward: the backward pass, a `plastax.policies.Backward` given the
            same way, or None to elide it.
        loss: the loss policy, or None to elide it.
        update: the update phase, a `plastax.policies.Update` with a
            connection half, a unit half or both, or None.
        optimizer: an optimizer bundle (`plastax.optim.Optimizer`), or None.
            Its `conn_fields` and `unit_fields` are added to the network's
            columns, and its `update()` fills the update phase: a half it
            supplies must not also be set in `update`.
        prune: the pruning policies, a `plastax.policies.Prune`, or None. Its
            `unit` half requires `unit_capacity`.
        grow: the growth policies, a `plastax.policies.Grow`, or None. Its
            `unit` half requires `unit_capacity`.
        reset_global: the global-state reset policy, or None to elide it.
        extra_unit_fields: the network's own per-unit fields beyond the
            builtin ones. An optimizer's are not listed here.
        extra_conn_fields: the network's own per-connection fields beyond the
            builtin ones.
        propagation: the propagation strategy used to schedule updates.
        kahn_max_depth: max depth for Kahn-order propagation, or None if unbounded.
        sharding: Scheme-A sharding config, or None for a single device.
        seed: the network seed keying the framework's counter-based RNG
            (`plastax.rng`); identical seeds give identical draw streams.
        structural_interval: run the structural phases (unit addition and
            growth, and so the resort growth triggers) only every this many
            steps -- ``step % n == 0`` fires them. Default 1 (every step).
            The growth rule's own ``trigger`` composes on top: both gates
            must pass for growth to run. Pruning is not gated: unit and
            connection pruning run every step, as in plastax-cpp.
        unit_capacity: the number of unit slots, or None (the default) for
            exactly the built unit count. A capacity sizes every unit column
            to that many slots and adds the built-in `PRUNED` column: the
            built units are live and the slots above them are free (marked
            pruned). A slot that holds no live unit is skipped by every pass's
            apply and by connection growth, and keeps its field defaults.
            Input and output units are always built units and are never
            pruned.
        capacity_headroom: extra room for connections in every bucket of a
            network built from its topology, beyond what its layers are given,
            as a fraction of that: 0.0 (the default) for none, 1.0 for as much
            again. It belongs to no layer.
        capacity_align: the number of connection slots a bucket's capacity is
            rounded up to a multiple of, for a network built from its
            topology. A power of two; 1024 by default.
        max_levels: the unit-level bound: a child unit's level is clamped to
            ``[1, max_levels - 1]``. Default 1024, the C++ library's bound.
        batch_reduction: how a batched step combines the per-sample unit
            columns before its once-per-batch phases (see `BatchReductionPolicy`),
            or None (the default). A batched step requires one; a streaming
            step ignores it.
    """

    topology: Module | None = None
    input_shape: tuple[int, ...] | None = None
    globals: GS | None = None

    forward: Forward[GS]
    backward: Backward[GS] | None = None
    loss: LossPolicy[GS] | None = None
    update: Update[GS] | None = None
    optimizer: Optimizer | None = None
    prune: Prune[GS] | None = None
    grow: Grow[GS] | None = None
    reset_global: ResetGlobalPolicy[GS] | None = None

    extra_unit_fields: tuple[FieldSpec[np.generic], ...] = ()
    extra_conn_fields: tuple[FieldSpec[np.generic], ...] = ()
    propagation: Propagation = Propagation.TOPOLOGICAL
    kahn_max_depth: int | None = None
    sharding: ShardSpec | None = None
    seed: int = 0
    structural_interval: int = 1
    unit_capacity: int | None = None
    capacity_headroom: float = 0.0
    capacity_align: int = 1024
    max_levels: int = 1024
    batch_reduction: BatchReductionPolicy | None = None

    def __init_subclass__(cls) -> None:
        """Validate the trait slots when a Network subclass is defined."""
        _validate_traits(cls)

    @classmethod
    def resolved(cls) -> Resolved[GS]:
        """The optional policies and the fields, as the engine reads them.

        `update`, `prune` and `grow` are pairs that may be absent or half set,
        and the optimizer adds to the update and to the fields. This gives
        every cell, None where nothing is declared.

        Returns:
            The resolved policies and field lists.

        Raises:
            TypeError: if `optimizer` and `update` both supply the same half.
        """
        opt = cls.optimizer
        from_opt = None if opt is None else opt.update()
        halves: dict[str, Any] = {"conn": None, "unit": None}
        for pair in (from_opt, cls.update):
            if pair is None:
                continue
            for half in ("conn", "unit"):
                policy = _on_network(getattr(pair, half))
                if policy is None:
                    continue
                if halves[half] is not None:
                    raise TypeError(
                        f"{cls.__name__}: the {half} update is given by both "
                        f"`optimizer` and `update`; a network has one {half} "
                        f"update"
                    )
                halves[half] = policy
        prune, grow = cls.prune, cls.grow
        # Narrowed here and not through `_on_network`: mypy joins the two growth
        # protocols to `object` when it solves that function's type variable.
        grow_conn = (
            None if grow is None or isinstance(grow.conn, Inherit) else grow.conn
        )
        return Resolved(
            update_conn=halves["conn"],
            update_unit=halves["unit"],
            prune_conn=None if prune is None else _on_network(prune.conn),
            prune_unit=None if prune is None else _on_network(prune.unit),
            grow_conn=grow_conn,
            grow_unit=None if grow is None else _on_network(grow.unit),
            unit_fields=(
                *cls.extra_unit_fields,
                *(() if opt is None else opt.unit_fields),
            ),
            conn_fields=(
                *cls.extra_conn_fields,
                *(() if opt is None else opt.conn_fields),
            ),
        )

    @classmethod
    def build(
        cls, key: PRNGKeyArray, *, globals: GS | None = None
    ) -> tuple[NetworkStatic, NetworkState[GS]]:
        """Trace the class's topology and materialize it into a state.

        The entry point of the functional layer: the pair is what `make_step`
        and `Driver` take. `init` is the usual way to run a network.

        The topology is traced with `input_shape` given to the sole argument of
        its `__call__`, and the record is materialized with
        `plastax.layers.materialize`: the weights are drawn from `key`, each
        block from a key derived from its path.

        Args:
            key: the key the weights are drawn from.
            globals: the value the global state starts from, in place of the
                class's `globals`. None uses the class's.

        Returns:
            The `(static, state)` pair. `static.input_shape` and
            `static.output_shape` hold the shapes a step's inputs and targets
            are given in.

        Raises:
            BuildError: if the class has no `topology` or no `input_shape`, if
                the topology does not take exactly one handle, if the policies
                have a fault (see `faults`), if a layer is given policies of
                its own (which the engine cannot run yet), or if the globals
                are not a pytree of numbers and arrays. Also whatever `trace`
                and `materialize` raise.
            NotImplementedError: if the topology returns more than one handle.
        """
        # Imported here: plastax.layers imports this module.
        from plastax.layers import BuildError, check, materialize, resolve

        name = cls.__name__
        record = cls._trace()
        if len(record.outputs) != 1:
            raise NotImplementedError(
                f"the topology of {name} returns {len(record.outputs)} handles; "
                f"only one is supported so far"
            )
        # Read as objects: a subclass may have set anything.
        headroom: object = cls.capacity_headroom
        align: object = cls.capacity_align
        if isinstance(headroom, bool) or not isinstance(headroom, int | float):
            raise BuildError(
                f"{name}.capacity_headroom is a fraction of extra room, a number "
                f"of at least 0; got {headroom!r}"
            )
        if headroom < 0:
            raise BuildError(
                f"{name}.capacity_headroom is a fraction of extra room and cannot "
                f"be negative; got {headroom!r}"
            )
        if (
            isinstance(align, bool)
            or not isinstance(align, int)
            or align < 1
            or align & (align - 1)
        ):
            raise BuildError(
                f"{name}.capacity_align is the number of slots a bucket is "
                f"rounded up to a multiple of, a power of two such as 1024; got "
                f"{align!r}"
            )
        on_device = cls._globals_on_device(globals)
        table = resolve(cls, record)
        # What can be fixed is reported before what cannot be run yet.
        faults = check(cls, record, table, on_device)
        if faults:
            count = "1 fault" if len(faults) == 1 else f"{len(faults)} faults"
            raise BuildError(
                f"{name} has {count} in its policies.\n\n"
                + "\n\n".join(str(fault) for fault in faults)
                + f"\n\n`{name}.faults()` returns these as a list."
            )
        _refuse_layer_policies(cls, table)
        return materialize(
            record,
            cls,
            key,
            globals_=on_device,
            capacity_headroom=float(headroom),
            capacity_align=align,
        )

    @classmethod
    def faults(cls) -> list[Fault]:
        """Every fault in the network's policies, found without building it.

        Traces the topology, resolves each layer's policies against the
        network's, and traces every policy once with abstract values: a `map`
        against the `combine` that reduces it, and every write against the field
        it is written to. Nothing is materialized, so it needs no key. `build`
        runs the same checks and raises on the first non-empty result.

        A `BuildError` is raised if the class has no `topology` or no
        `input_shape`, the topology does not take exactly one handle, or the
        class's `globals` are not a pytree of numbers and arrays. Also whatever
        `trace` raises.

        Returns:
            The faults, empty when the policies are sound.
        """
        # Imported here: plastax.layers imports this module.
        from plastax.layers import check, resolve

        record = cls._trace()
        return check(cls, record, resolve(cls, record), cls._globals_on_device(None))

    @classmethod
    def _globals_on_device(cls, globals: GS | None) -> Any:
        """The global state a network starts from, as arrays.

        Args:
            globals: the value to use in place of the class's `globals`, or
                None for the class's.

        Returns:
            The value with every leaf an array.

        Raises:
            BuildError: if the value is not a pytree of numbers and arrays.
        """
        # Imported here: plastax.layers imports this module.
        from plastax.layers import BuildError

        value = cls.globals if globals is None else globals
        try:
            return jax.tree_util.tree_map(jnp.asarray, value)
        except TypeError as error:
            raise BuildError(
                f"the globals of {cls.__name__} are not a pytree of numbers and "
                f"arrays: {error}. A dataclass is made one with "
                f"jax.tree_util.register_dataclass"
            ) from error

    @classmethod
    def policy_table(cls) -> PolicyTable:
        """The policy of every layer in every phase, as the network resolves it.

        Traces the topology and resolves each block's policies against the
        network's: a layer's own where it was given one, the network's
        otherwise. Nothing is materialized, so it needs no key and works for a
        network the engine cannot run yet. `print(Net.policy_table())` gives a
        Markdown table.

        A `BuildError` is raised if the class has no `topology` or no
        `input_shape`, or the topology does not take exactly one handle. Also
        whatever `trace` raises.

        Returns:
            The resolved table.
        """
        # Imported here: plastax.layers imports this module.
        from plastax.layers import resolve

        return resolve(cls, cls._trace())

    @classmethod
    def _trace(cls) -> Record:
        """Trace the class's topology with its `input_shape`.

        Returns:
            The build record.

        Raises:
            BuildError: if the class has no `topology` or no `input_shape`, or
                the topology does not take exactly one handle. Also whatever
                `trace` raises.
        """
        # Imported here: plastax.layers imports this module.
        from plastax.layers import BuildError, trace

        name = cls.__name__
        if cls.topology is None:
            raise BuildError(
                f"{name} has no topology to build from; set `topology` and "
                f"`input_shape` on the class, or build the network with "
                f"NetworkBuilder"
            )
        if cls.input_shape is None:
            raise BuildError(
                f"{name} has a topology and no `input_shape`; the shape of the "
                f"input units is needed to trace it"
            )
        parameters = (
            list(inspect.signature(cls.topology.__call__).parameters)
            if callable(cls.topology)
            else []
        )
        if len(parameters) != 1:
            raise BuildError(
                f"{name}.input_shape describes one input, and the topology "
                f"{type(cls.topology).__name__}.__call__ takes {len(parameters)} "
                f"arguments {parameters}; a topology with several inputs is not "
                f"supported yet"
            )
        shape = tuple(int(n) for n in cls.input_shape)
        return trace(cls.topology, {parameters[0]: shape})

    @classmethod
    def init(cls, key: PRNGKeyArray, *, globals: GS | None = None) -> Driver[GS]:
        """Build the network and return the `Driver` that runs it.

        The driver owns the network's state and static configuration, and runs
        it one step at a time (`Driver.step`) or a chunk of steps on the device
        (`Driver.scan`), regrowing and resorting as the network needs.

        Args:
            key: the key the weights are drawn from.
            globals: the value the global state starts from, in place of the
                class's `globals`. None uses the class's.

        Returns:
            A driver holding the built network.
        """
        # Imported here: plastax.driver imports this module.
        from plastax.driver import Driver

        static, state = cls.build(key, globals=globals)
        return Driver(cls, static, state)


def _on_network[P](half: P | Inherit | None) -> P | None:
    """Read a half of a network's own pair: nothing to inherit, so none.

    Args:
        half: the half as the pair holds it.

    Returns:
        The policy, or None where the half is None or was not given.
    """
    return None if isinstance(half, Inherit) else half


def _refuse_layer_policies(cls: type[Network[Any]], table: PolicyTable) -> None:
    """Refuse to build a network whose layers differ from it in a policy.

    The engine runs one set of policies for the whole network. Until it runs a
    policy on the elements of one layer, a network that needs that is refused
    here, by name, and not run with the wrong policies.

    Args:
        cls: the network class being built.
        table: its resolved policies.

    Raises:
        BuildError: if any block's policy differs from the network's.
    """
    # Imported here: plastax.layers imports this module.
    from plastax.layers import BuildError
    from plastax.layers._module import _show

    overrides = table.overrides()
    if not overrides:
        return

    def named(policy: object) -> str:
        return "none" if policy is None else type(policy).__name__

    lines: list[str] = []
    width = max(len(named(o.policy)) for o in overrides)
    last: tuple[tuple[str, ...], str] | None = None
    for o in overrides:
        if (o.path, o.kind) != last:
            lines.append(f"  {_show(o.path)}  ({o.kind})")
            last = (o.path, o.kind)
        lines.append(
            f"      {o.phase:<9}{named(o.policy):<{width}}   "
            f"the network has {named(o.network)}"
        )
    name = cls.__name__
    raise BuildError(
        f"{name} cannot be run yet. Its layers are given policies of their "
        f"own, and the engine runs one set of policies for the whole network "
        f"(layer policies, stage 3).\n\n"
        + "\n".join(lines)
        + f"\n\nThe declaration itself is valid. `print({name}.policy_table())` "
        f"shows what every layer resolved to."
    )


_RESERVED_FIELD_NAMES = frozenset(
    {
        FROM_ID.name,
        TO_ID.name,
        DEAD.name,
        WEIGHT.name,
        ACTIVATION.name,
        LEVEL.name,
        PRUNED.name,
    }
)


def _validate_monoid_tree(
    tree: object, cls: type[Network[Any]], attr_name: str
) -> None:
    """Recursively check that `tree` is a well-formed MonoidTree.

    A well-formed MonoidTree is a Monoid leaf, or a non-empty dict/tuple of
    well-formed MonoidTrees -- a product of monoids is itself a monoid.

    Args:
        tree: the candidate MonoidTree to validate.
        cls: the Network subclass being validated, used for error messages.
        attr_name: the name of the trait attribute `tree` belongs to.

    Raises:
        ValueError: if a dict or tuple node in `tree` is empty.
        TypeError: if `tree` is not a Monoid, dict, or tuple.
    """
    if isinstance(tree, Monoid):
        return
    if isinstance(tree, dict):
        if not tree:
            raise ValueError(
                f"{cls.__name__}.{attr_name}.combine: dict MonoidTree is empty"
            )
        for leaf in tree.values():
            _validate_monoid_tree(leaf, cls, attr_name)
        return
    if isinstance(tree, tuple):
        if not tree:
            raise ValueError(
                f"{cls.__name__}.{attr_name}.combine: tuple MonoidTree is empty"
            )
        for leaf in tree:
            _validate_monoid_tree(leaf, cls, attr_name)
        return
    raise TypeError(
        f"{cls.__name__}.{attr_name}.combine is not a well-formed MonoidTree "
        f"(Monoid | dict[str, MonoidTree] | tuple[MonoidTree, ...]); got {tree!r}"
    )


def _as_pair(
    cls: type[Network[Any]],
    name: str,
    pair_type: type,
    conn_trait: type,
    unit_trait: type,
) -> Any:
    """Read a pass field as its pair, or None when it is not set.

    Args:
        cls: the Network subclass being validated.
        name: "forward" or "backward".
        pair_type: `Forward` or `Backward`.
        conn_trait: the connection half's protocol.
        unit_trait: the unit half's protocol.

    Returns:
        The pair: the field itself, or the pair with the field's one object in
        both halves. None when the field is not set.

    Raises:
        TypeError: if the field is neither a pair nor one object with both
            halves, or a half of the pair does not satisfy its protocol.
    """
    value = getattr(cls, name, None)
    if value is None:
        return None
    if not isinstance(value, pair_type):
        if isinstance(value, conn_trait) and isinstance(value, unit_trait):
            return pair_type(conn=value, unit=value)
        raise TypeError(
            f"{cls.__name__}.{name} must be a plastax.policies."
            f"{pair_type.__name__}(conn=, unit=), or one object with map, "
            f"combine and apply; got {value!r}"
        )
    for half, trait, members in (
        ("conn", conn_trait, "map"),
        ("unit", unit_trait, "combine, apply"),
    ):
        if getattr(value, half) is INHERIT or getattr(value, half) is None:
            said = "is None" if getattr(value, half) is None else "was not given"
            raise TypeError(
                f"{cls.__name__}.{name} has no `{half}` policy (it {said}). A "
                f"network's {name} pass needs both halves, conn "
                f"({conn_trait.__name__}) and unit ({unit_trait.__name__}): a "
                f"network has nothing to inherit the other from. Only a layer "
                f"may give one half"
            )
        if not isinstance(getattr(value, half), trait):
            raise TypeError(
                f"{cls.__name__}.{name}.{half} must satisfy {trait.__name__} "
                f"({members}); got {getattr(value, half)!r}"
            )
    return value


def _warn_unused_linear_input(
    cls: type[Network[Any]], which: str, conn_half: object, unit_half: object
) -> None:
    """Warn when a connection half's `linear_input` cannot be used.

    Args:
        cls: the Network subclass being validated, for the message.
        which: "forward" or "backward".
        conn_half: the connection half of the pass.
        unit_half: the unit half of the same pass.
    """
    if getattr(conn_half, "linear_input", None) is None:
        return
    combine = getattr(unit_half, "combine", None)
    if any(combine is semiring.add for semiring in LINEAR_SEMIRINGS):
        return
    warnings.warn(
        f"{cls.__name__}.{which}.conn declares linear_input, and "
        f"{which}.unit.combine is {combine!r}, which is not the additive monoid "
        f"of a semiring the engine has a sparse product for "
        f"({', '.join(str(s) for s in LINEAR_SEMIRINGS)}, as (add, mul)); the "
        f"declaration is ignored and the pass runs per connection",
        stacklevel=4,
    )


def _validate_loss(cls: type[Network[Any]], loss: object) -> None:
    """Check the loss policy's shape and its declared seed field.

    Args:
        cls: the Network subclass being validated.
        loss: the candidate loss policy.

    Raises:
        TypeError: if `loss` still has the per-output signature, does not
            satisfy LossPolicy, or declares a seed field that is not a float unit
            column of `cls`.
    """
    if hasattr(loss, "per_output") and not hasattr(loss, "calculate_loss"):
        raise TypeError(
            f"{cls.__name__}.loss: the per-output loss signature was replaced by "
            "the whole-output one. Declare `seed_field` (the unit column the "
            "gradient seed is written to) and implement "
            "`calculate_loss(u, outputs, targets, g) -> (loss, seed)`."
        )
    if not isinstance(loss, LossPolicy):
        raise TypeError(
            f"{cls.__name__}.loss must satisfy LossPolicy (seed_field, "
            f"calculate_loss); got {loss!r}"
        )
    seed: object = loss.seed_field
    unit_fields = (ACTIVATION, *cls.resolved().unit_fields)
    if not isinstance(seed, FieldSpec) or seed not in unit_fields:
        raise TypeError(
            f"{cls.__name__}.loss.seed_field must be one of the network's unit "
            f"columns (ACTIVATION, an extra_unit_fields entry or an optimizer "
            f"field); got {seed!r}"
        )
    if not np.issubdtype(seed.dtype, np.floating):
        raise TypeError(
            f"{cls.__name__}.loss.seed_field {seed.name!r} must be a float "
            f"column; got dtype {seed.dtype}"
        )


def _validate_field_names(cls: type[Network[Any]]) -> None:
    """Check field-name uniqueness and reject reserved builtin names.

    Verifies uniqueness across builtin and extra unit/conn fields, and that
    user extra fields do not collide with reserved builtin names.

    Args:
        cls: the Network subclass whose extra fields are validated.

    Raises:
        ValueError: if an extra field name collides with a reserved builtin
            name, or if two extra field names collide with each other.
    """
    resolved = cls.resolved()
    all_extra = (*resolved.unit_fields, *resolved.conn_fields)
    seen: set[str] = set()
    for spec in all_extra:
        if spec.name in _RESERVED_FIELD_NAMES:
            raise ValueError(
                f"{cls.__name__}: extra field {spec.name!r} collides with a reserved "
                f"builtin name ({sorted(_RESERVED_FIELD_NAMES)})"
            )
        if spec.name in seen:
            raise ValueError(
                f"{cls.__name__}: duplicate extra field name {spec.name!r}"
            )
        seen.add(spec.name)


def _validate_traits(cls: type[Network[Any]]) -> None:
    """Run the runtime concept check for a Network subclass.

    Checks protocol conformance of each configured trait, delegating monoid
    tree structure and field-name checks to `_validate_monoid_tree` and
    `_validate_field_names` respectively.

    Args:
        cls: the Network subclass to validate.

    Raises:
        TypeError: if `forward` is missing, a pass is neither a pair nor one
            object with both halves, a configured policy does not satisfy its
            protocol, or `optimizer` and `update` supply the same half.
        ValueError: if a delegated check fails -- a malformed combine
            MonoidTree, or a field-name collision or duplicate.
    """
    if "neighbourhood" in vars(cls):
        raise TypeError(
            f"{cls.__name__}.neighbourhood is no longer a Network attribute: "
            "the growth window moved onto the growth rule. Set "
            "`max_level_gap` (int, default 1) on the grow.conn policy instead."
        )
    forward = _as_pair(cls, "forward", Forward, ForwardConnPolicy, ForwardUnitPolicy)
    if forward is None:
        raise TypeError(f"{cls.__name__}.forward is required and was not set")
    cls.forward = forward
    _validate_monoid_tree(forward.unit.combine, cls, "forward.unit")
    _warn_unused_linear_input(cls, "forward", forward.conn, forward.unit)
    backward = _as_pair(
        cls, "backward", Backward, BackwardConnPolicy, BackwardUnitPolicy
    )
    cls.backward = backward
    if backward is not None:
        _validate_monoid_tree(backward.unit.combine, cls, "backward.unit")
        _warn_unused_linear_input(cls, "backward", backward.conn, backward.unit)

    for name, pair_type in (("update", Update), ("prune", Prune), ("grow", Grow)):
        pair = getattr(cls, name)
        if pair is not None and not isinstance(pair, pair_type):
            raise TypeError(
                f"{cls.__name__}.{name} must be a plastax.policies."
                f"{pair_type.__name__} or None; got {pair!r}"
            )
    if cls.optimizer is not None and not isinstance(cls.optimizer, Optimizer):
        raise TypeError(
            f"{cls.__name__}.optimizer must satisfy plastax.optim.Optimizer "
            f"(conn_fields, unit_fields, update); got {cls.optimizer!r}"
        )
    resolved = cls.resolved()
    slots: tuple[tuple[str, object, type, str], ...] = (
        ("update conn", resolved.update_conn, UpdateConnPolicy, "incoming, outgoing"),
        ("update unit", resolved.update_unit, UpdateUnitPolicy, "update"),
        ("prune.conn", resolved.prune_conn, PruneConnPolicy, "predicate"),
        ("prune.unit", resolved.prune_unit, PruneUnitPolicy, "predicate"),
        ("grow.unit", resolved.grow_unit, AddUnitPolicy, "spawn, init"),
    )
    for where, policy, trait, members in slots:
        if policy is not None and not isinstance(policy, trait):
            raise TypeError(
                f"{cls.__name__}: the {where} policy must satisfy "
                f"{trait.__name__} ({members}); got {policy!r}"
            )

    if cls.loss is not None:
        _validate_loss(cls, cls.loss)

    interval: object = getattr(cls, "structural_interval", 1)
    if not isinstance(interval, int) or isinstance(interval, bool) or interval < 1:
        raise TypeError(
            f"{cls.__name__}.structural_interval must be an int >= 1; got {interval!r}"
        )

    _validate_unit_slots(cls)

    grow_conn = resolved.grow_conn
    if grow_conn is not None:
        grid = isinstance(grow_conn, ScoreAddConnPolicy)
        proposed = isinstance(grow_conn, ProposeAddConnPolicy)
        if grid == proposed:
            raise TypeError(
                f"{cls.__name__}.grow.conn must satisfy exactly one of "
                f"ScoreAddConnPolicy (score) or ProposeAddConnPolicy (propose); "
                f"got {grow_conn!r}"
            )
        if isinstance(grow_conn, ProposeAddConnPolicy):
            _validate_propose_rule(cls, grow_conn)
        else:
            _validate_score_rule(cls, grow_conn)
        _validate_growth_knobs(cls, grow_conn)

    if cls.reset_global is not None and not isinstance(
        cls.reset_global, ResetGlobalPolicy
    ):
        raise TypeError(
            f"{cls.__name__}.reset_global must satisfy ResetGlobalPolicy; "
            f"got {cls.reset_global!r}"
        )

    _validate_field_names(cls)
    _validate_batch_reduction(cls)


def _validate_batch_reduction(cls: type[Network[Any]]) -> None:
    """Check the batch-reduction policy against the network's unit columns.

    Every unit column's reduction must suit its dtype (MEAN a floating column,
    SUM a numeric non-boolean one), `LEVEL` and `PRUNED` may only be
    NOT_BATCHED or undeclared, a `FieldReductions` may declare only the
    network's own columns, and the columns every batched step writes per
    sample -- `ACTIVATION` (the input scatter) and the loss's `seed_field` --
    must be declared MEAN, SUM or FIRST. The columns the rules write are
    checked when a batched step is first traced (see `make_step`).

    Args:
        cls: the Network subclass being validated.

    Raises:
        TypeError: if `batch_reduction` does not satisfy BatchReductionPolicy, or a
            reduction does not suit its column's dtype.
        ValueError: if the policy declares a column the network does not have,
            reduces `LEVEL` or `PRUNED`, or leaves a column every batched step
            writes undeclared or NOT_BATCHED.
    """
    policy: object = getattr(cls, "batch_reduction", None)
    if policy is None:
        return
    if not isinstance(policy, BatchReductionPolicy):
        raise TypeError(
            f"{cls.__name__}.batch_reduction must satisfy BatchReductionPolicy "
            f"(reduction); got {policy!r}"
        )
    unit_fields: tuple[FieldSpec[Any], ...] = (
        ACTIVATION,
        LEVEL,
        *((PRUNED,) if cls.unit_capacity is not None else ()),
        *cls.extra_unit_fields,
    )
    if isinstance(policy, FieldReductions):
        known = {spec.name for spec in unit_fields}
        for spec, _ in policy.fields:
            if spec.name not in known:
                raise ValueError(
                    f"{cls.__name__}.batch_reduction declares {spec.name!r}, "
                    f"which is not a unit column of the network ({sorted(known)})"
                )
    for spec in unit_fields:
        red: object = policy.reduction(spec)
        if red is None or red is Reduction.NOT_BATCHED:
            continue
        if not isinstance(red, Reduction):
            raise TypeError(
                f"{cls.__name__}.batch_reduction.reduction({spec.name!r}) must "
                f"return a Reduction or None; got {red!r}"
            )
        if spec.name in _STRUCTURAL_UNIT_FIELDS:
            raise ValueError(
                f"{cls.__name__}.batch_reduction: {spec.name!r} is written only "
                "by the structural phases; declare it NOT_BATCHED or leave it "
                f"undeclared, not {red.name}"
            )
        if red is Reduction.MEAN and not np.issubdtype(spec.dtype, np.floating):
            raise TypeError(
                f"{cls.__name__}.batch_reduction: MEAN needs a floating column; "
                f"{spec.name!r} is {spec.dtype}"
            )
        if red is Reduction.SUM and (
            spec.dtype == np.bool_ or not np.issubdtype(spec.dtype, np.number)
        ):
            raise TypeError(
                f"{cls.__name__}.batch_reduction: SUM needs a numeric, "
                f"non-boolean column; {spec.name!r} is {spec.dtype}"
            )
    try:
        batch_written_reduction(policy, ACTIVATION, where="the input scatter")
        if cls.loss is not None:
            batch_written_reduction(
                policy, cls.loss.seed_field, where="the loss's seed_field"
            )
    except ValueError as err:
        raise ValueError(f"{cls.__name__}.{err}") from None


def _validate_unit_slots(cls: type[Network[Any]]) -> None:
    """Check `unit_capacity`, `max_levels` and the unit lifecycle slots.

    Args:
        cls: the Network subclass being validated (for error messages).

    Raises:
        TypeError: if `unit_capacity` is neither None nor an int >= 1,
            `max_levels` is not an int >= 2, or `prune.unit` or `grow.unit` is
            declared without a `unit_capacity`.
    """
    capacity: object = getattr(cls, "unit_capacity", None)
    if capacity is not None and (
        not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1
    ):
        raise TypeError(
            f"{cls.__name__}.unit_capacity must be None or an int >= 1; "
            f"got {capacity!r}"
        )
    if cls.resolved().prune_unit is not None and capacity is None:
        raise TypeError(
            f"{cls.__name__}.prune.unit requires a unit_capacity: the `PRUNED` "
            "column that marks pruned units exists only with one"
        )
    if cls.resolved().grow_unit is not None and capacity is None:
        raise TypeError(
            f"{cls.__name__}.grow.unit requires a unit_capacity: children are "
            "placed in the free unit slots, which exist only with one"
        )
    reject_sharded_unit_lifecycle(cls, cls.sharding)
    max_levels: object = getattr(cls, "max_levels", 1024)
    if (
        not isinstance(max_levels, int)
        or isinstance(max_levels, bool)
        or max_levels < 2
    ):
        raise TypeError(
            f"{cls.__name__}.max_levels must be an int >= 2; got {max_levels!r}"
        )


def reject_sharded_unit_lifecycle(
    net: type[Network[Any]], sharding: ShardSpec | None
) -> None:
    """Refuse a network that prunes or adds units under Scheme-A sharding.

    Units are replicated, but pruning one tombstones its connections in the
    sharded connection arenas, and placing a child in a free slot raises the
    same question of how the shards stay balanced; neither path is designed
    yet.

    Args:
        net: the Network subclass (for the check and error messages).
        sharding: the sharding the network is defined or built with.

    Raises:
        NotImplementedError: if `net` prunes or adds units and
            `sharding` is set.
    """
    if sharding is None:
        return
    if net.resolved().prune_unit is not None:
        raise NotImplementedError(
            f"{net.__name__}: unit pruning under sharding is not supported yet"
        )
    if net.resolved().grow_unit is not None:
        raise NotImplementedError(
            f"{net.__name__}: unit addition under sharding is not supported yet"
        )


_PROPOSE_ARITY = {
    # positional parameters of `propose` after self: (names, shape hint)
    "per_unit": (5, "propose(self, u, i, j, g, rng)"),
    "per_connection": (6, "propose(self, u, c, cid, j, g, rng)"),
    "global": (4, "propose(self, u, j, g, rng)"),
}


def _validate_propose_rule(
    cls: type[Network[Any]], ac: ProposeAddConnPolicy[Any]
) -> None:
    """Check a ProposeAddConnPolicy's knobs and `propose` shape at class definition.

    Args:
        cls: the Network subclass being validated (for error messages).
        ac: the declared propose rule.

    Raises:
        TypeError: on a removed attribute (`num_proposals`, `dedupe`), a bad
            `proposer` / `proposals_per_proposer`, or a `propose` whose
            positional arity does not match the declared proposer.
    """
    if hasattr(ac, "num_proposals"):
        raise TypeError(
            f"{cls.__name__}.grow.conn.num_proposals was renamed: declare "
            "`proposals_per_proposer` (int >= 1) on the propose rule instead."
        )
    n: object = getattr(ac, "proposals_per_proposer", None)
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise TypeError(
            f"{cls.__name__}.grow.conn.proposals_per_proposer must be an "
            f"int >= 1; got {n!r}"
        )
    proposer = getattr(ac, "proposer", "per_unit")
    if not isinstance(proposer, str) or proposer not in _PROPOSE_ARITY:
        raise TypeError(
            f"{cls.__name__}.grow.conn.proposer must be one of "
            f"{sorted(_PROPOSE_ARITY)}; got {proposer!r}"
        )
    want, shape = _PROPOSE_ARITY[proposer]
    sig = inspect.signature(ac.propose)
    if any(
        prm.kind is inspect.Parameter.VAR_POSITIONAL for prm in sig.parameters.values()
    ):
        return  # *args adapters are arity-unverifiable; the call site decides
    got = sum(
        1
        for prm in sig.parameters.values()
        if prm.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    if got != want:
        raise TypeError(
            f"{cls.__name__}.grow.conn: a {proposer!r} proposer's propose "
            f"takes {want} arguments after self -- expected `{shape}`, got {got}."
        )


_SELECTIONS = ("top_k", "threshold", "all")
_DIRECTIONS = ("any", "deeper", "same_or_deeper")
_CANDIDATES = ("exhaustive", "shortlist", "shortlist_per_level")
_LEVEL_CAP_SCOPES = ("source_level", "step")


def _validate_growth_knobs(cls: type[Network[Any]], ac: object) -> None:
    """Check the growth-rule knobs shared by both strategies.

    Args:
        cls: the Network subclass being validated (for error messages).
        ac: the declared growth rule.

    Raises:
        TypeError: on a removed attribute name, an unknown knob value, a
            missing required companion (`max_new_per_level`, `threshold`,
            `when`), or a mis-typed knob.
    """
    for removed, repl in (
        ("dedupe", "set `dedupe_live` and/or `dedupe_step` (bool, default False)"),
        ("max_candidates", "declare `max_new_per_level` (int >= 1)"),
        (
            "max_candidate_units",
            'declare `candidates = "shortlist"` and `shortlist_size` (int >= 1)',
        ),
        ("shortlist_per_level", 'declare `candidates = "shortlist_per_level"`'),
    ):
        if hasattr(ac, removed):
            raise TypeError(
                f"{cls.__name__}.grow.conn.{removed} was removed: {repl} "
                "on the growth rule instead."
            )
    selection = getattr(ac, "selection", "top_k")
    if selection not in _SELECTIONS:
        raise TypeError(
            f"{cls.__name__}.grow.conn.selection must be one of "
            f"{_SELECTIONS}; got {selection!r}"
        )
    mnpl: object = getattr(ac, "max_new_per_level", None)
    if selection != "all" or mnpl is not None:
        if not isinstance(mnpl, int) or isinstance(mnpl, bool) or mnpl < 1:
            raise TypeError(
                f"{cls.__name__}.grow.conn.max_new_per_level must be an "
                f'int >= 1 (required unless selection = "all"); got {mnpl!r}'
            )
    if selection == "threshold" and not callable(getattr(ac, "threshold", None)):
        raise TypeError(
            f'{cls.__name__}.grow.conn: selection = "threshold" requires a '
            "`threshold(g)` method on the rule."
        )
    scope = getattr(ac, "level_cap_scope", "source_level")
    if scope not in _LEVEL_CAP_SCOPES:
        raise TypeError(
            f"{cls.__name__}.grow.conn.level_cap_scope must be one of "
            f"{_LEVEL_CAP_SCOPES}; got {scope!r}"
        )
    mnps: object = getattr(ac, "max_new_per_step", None)
    if mnps is not None and (
        not isinstance(mnps, int) or isinstance(mnps, bool) or mnps < 1
    ):
        raise TypeError(
            f"{cls.__name__}.grow.conn.max_new_per_step must be an int >= 1 "
            f"or None; got {mnps!r}"
        )
    direction = getattr(ac, "direction", "any")
    if direction not in _DIRECTIONS:
        raise TypeError(
            f"{cls.__name__}.grow.conn.direction must be one of "
            f"{_DIRECTIONS}; got {direction!r}"
        )
    trigger: object = getattr(ac, "trigger", "every_step")
    ok = trigger in ("every_step", "on_units_added", "when") or (
        isinstance(trigger, tuple)
        and len(trigger) == 2
        and trigger[0] == "every"
        and isinstance(trigger[1], int)
        and not isinstance(trigger[1], bool)
        and trigger[1] >= 1
    )
    if not ok:
        raise TypeError(
            f"{cls.__name__}.grow.conn.trigger must be 'every_step', "
            f"('every', n >= 1), 'on_units_added' or 'when'; got {trigger!r}"
        )
    if trigger == "when" and not callable(getattr(ac, "when", None)):
        raise TypeError(
            f"{cls.__name__}.grow.conn: trigger = 'when' requires a "
            "`when(g)` method on the rule."
        )
    on_overflow = getattr(ac, "on_overflow", "flag")
    if on_overflow not in ("flag", "error"):
        raise TypeError(
            f"{cls.__name__}.grow.conn.on_overflow must be 'flag' or "
            f"'error'; got {on_overflow!r}"
        )
    for flag in ("dedupe_live", "dedupe_step", "allow_self_loops"):
        v = getattr(ac, flag, False)
        if not isinstance(v, bool):
            raise TypeError(
                f"{cls.__name__}.grow.conn.{flag} must be a bool; got {v!r}"
            )


def _validate_score_rule(cls: type[Network[Any]], ac: ScoreAddConnPolicy[Any]) -> None:
    """Check a ScoreAddConnPolicy's candidate-production knobs at class definition.

    Args:
        cls: the Network subclass being validated (for error messages).
        ac: the declared score rule.

    Raises:
        TypeError: on an unknown `candidates`, a shortlist without
            `shortlist_size` / `importance`, or a shortlist size that is not
            a positive int.
    """
    candidates = getattr(ac, "candidates", "exhaustive")
    if candidates not in _CANDIDATES:
        raise TypeError(
            f"{cls.__name__}.grow.conn.candidates must be one of "
            f"{_CANDIDATES}; got {candidates!r}"
        )
    if candidates != "exhaustive":
        size: object = getattr(ac, "shortlist_size", None)
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise TypeError(
                f"{cls.__name__}.grow.conn.shortlist_size must be an int >= 1 "
                f"for candidates = {candidates!r}; got {size!r}"
            )
        if not callable(getattr(ac, "importance", None)):
            raise TypeError(
                f"{cls.__name__}.grow.conn: candidates = {candidates!r} "
                "requires an `importance(u, i, g)` method on the rule."
            )
