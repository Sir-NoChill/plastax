"""The Network base class and its class-definition checks."""

from __future__ import annotations

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
from plastax.policies import (
    AddUnitPolicy,
    BackwardConnPolicy,
    BackwardUnitPolicy,
    BatchReductionPolicy,
    Conn,
    FieldReductions,
    ForwardConnPolicy,
    ForwardUnitPolicy,
    LossPolicy,
    ProposeAddConnPolicy,
    PruneConnPolicy,
    PruneUnitPolicy,
    Reduction,
    ResetGlobalPolicy,
    ScoreAddConnPolicy,
    Unit,
    UpdateConnPolicy,
    UpdateUnitPolicy,
    batch_written_reduction,
)
from plastax.policies._batch import _STRUCTURAL_UNIT_FIELDS

if TYPE_CHECKING:
    from plastax.driver import Driver
    from plastax.layers import Module
    from plastax.state import NetworkState, NetworkStatic


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
        conn_policy: the behaviour of the network's connections, a
            `plastax.policies.Conn`: the connection halves of the forward and
            backward passes, the update and the prune. A slot left None is
            elided.
        unit_policy: the behaviour of the network's units, a
            `plastax.policies.Unit`: the unit halves of the forward and
            backward passes, the update and the prune. Its `prune` requires
            `unit_capacity`.
        loss: the loss policy, or None to elide it.
        add_unit: the unit addition policy, or None to elide it. Requires
            `unit_capacity`.
        add_conn: the connection growth policy, or None to elide it.
        reset_global: the global-state reset policy, or None to elide it.
        extra_unit_fields: extra per-unit fields beyond the builtin ones.
        extra_conn_fields: extra per-connection fields beyond the builtin ones.
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

    conn_policy: Conn[GS]
    unit_policy: Unit[GS]
    loss: LossPolicy[GS] | None = None
    add_unit: AddUnitPolicy[GS] | None = None
    add_conn: ScoreAddConnPolicy[GS] | ProposeAddConnPolicy[GS] | None = None
    reset_global: ResetGlobalPolicy[GS] | None = None

    extra_unit_fields: tuple[FieldSpec[np.generic], ...] = ()
    extra_conn_fields: tuple[FieldSpec[np.generic], ...] = ()
    propagation: Propagation = Propagation.TOPOLOGICAL
    kahn_max_depth: int | None = None
    sharding: ShardSpec | None = None
    seed: int = 0
    structural_interval: int = 1
    unit_capacity: int | None = None
    max_levels: int = 1024
    batch_reduction: BatchReductionPolicy | None = None

    def __init_subclass__(cls) -> None:
        """Validate the trait slots when a Network subclass is defined."""
        _validate_traits(cls)

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
                the topology does not take exactly one handle, or if the
                globals are not a pytree of numbers and arrays. Also whatever
                `trace` and `materialize` raise.
            NotImplementedError: if the topology returns more than one handle.
        """
        # Imported here: plastax.layers imports this module.
        from plastax.layers import BuildError, materialize, trace

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
        record = trace(cls.topology, {parameters[0]: shape})
        if len(record.outputs) != 1:
            raise NotImplementedError(
                f"the topology of {name} returns {len(record.outputs)} handles; "
                f"only one is supported so far"
            )

        value = cls.globals if globals is None else globals
        try:
            on_device = jax.tree_util.tree_map(jnp.asarray, value)
        except TypeError as error:
            raise BuildError(
                f"the globals of {name} are not a pytree of numbers and arrays: "
                f"{error}. A dataclass is made one with "
                f"jax.tree_util.register_dataclass"
            ) from error
        return materialize(record, cls, key, globals_=on_device)

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
    if any(combine is m for m in LINEAR_SEMIRINGS.values()):
        return
    warnings.warn(
        f"{cls.__name__}.conn_policy.{which} declares linear_input, and "
        f"unit_policy.{which}.combine is {combine!r}, which is not the monoid of "
        f"a semiring the engine has a sparse product for "
        f"({', '.join(LINEAR_SEMIRINGS)}); the declaration is ignored and the "
        f"pass runs per connection",
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
    unit_fields = (ACTIVATION, *cls.extra_unit_fields)
    if not isinstance(seed, FieldSpec) or seed not in unit_fields:
        raise TypeError(
            f"{cls.__name__}.loss.seed_field must be one of the network's unit "
            f"columns (ACTIVATION or an extra_unit_fields entry); got {seed!r}"
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
    all_extra = (*cls.extra_unit_fields, *cls.extra_conn_fields)
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
        TypeError: if `conn_policy` or `unit_policy` is missing, a configured
            policy does not satisfy its protocol, or only one half of the
            backward pass is set.
        ValueError: if a delegated check fails -- a malformed combine
            MonoidTree, or a field-name collision or duplicate.
    """
    if "neighbourhood" in vars(cls):
        raise TypeError(
            f"{cls.__name__}.neighbourhood is no longer a Network attribute: "
            "the growth window moved onto the growth rule. Set "
            "`max_level_gap` (int, default 1) on the add_conn policy instead."
        )
    conn = getattr(cls, "conn_policy", None)
    unit = getattr(cls, "unit_policy", None)
    if not isinstance(conn, Conn):
        raise TypeError(
            f"{cls.__name__}.conn_policy is required and must be a "
            f"plastax.policies.Conn; got {conn!r}"
        )
    if not isinstance(unit, Unit):
        raise TypeError(
            f"{cls.__name__}.unit_policy is required and must be a "
            f"plastax.policies.Unit; got {unit!r}"
        )
    slots: tuple[tuple[str, object, type, str], ...] = (
        ("conn_policy.forward", conn.forward, ForwardConnPolicy, "map"),
        ("unit_policy.forward", unit.forward, ForwardUnitPolicy, "combine, apply"),
        ("conn_policy.backward", conn.backward, BackwardConnPolicy, "map"),
        ("unit_policy.backward", unit.backward, BackwardUnitPolicy, "combine, apply"),
        ("conn_policy.update", conn.update, UpdateConnPolicy, "incoming, outgoing"),
        ("unit_policy.update", unit.update, UpdateUnitPolicy, "update"),
        ("conn_policy.prune", conn.prune, PruneConnPolicy, "predicate"),
        ("unit_policy.prune", unit.prune, PruneUnitPolicy, "predicate"),
    )
    for where, policy, trait, members in slots:
        if policy is None and not where.endswith(".forward"):
            continue
        if not isinstance(policy, trait):
            raise TypeError(
                f"{cls.__name__}.{where} must satisfy {trait.__name__} "
                f"({members}); got {policy!r}"
            )
    if (conn.backward is None) != (unit.backward is None):
        present, absent = (
            ("conn_policy", "unit_policy")
            if unit.backward is None
            else ("unit_policy", "conn_policy")
        )
        raise TypeError(
            f"{cls.__name__}.{present}.backward is set and {absent}.backward is "
            f"not; a backward pass needs both halves, the connection's map and "
            f"the unit's combine and apply"
        )
    _validate_monoid_tree(unit.forward.combine, cls, "unit_policy.forward")
    _warn_unused_linear_input(cls, "forward", conn.forward, unit.forward)
    if unit.backward is not None:
        _validate_monoid_tree(unit.backward.combine, cls, "unit_policy.backward")
        _warn_unused_linear_input(cls, "backward", conn.backward, unit.backward)

    if cls.loss is not None:
        _validate_loss(cls, cls.loss)

    if cls.add_unit is not None and not isinstance(cls.add_unit, AddUnitPolicy):
        raise TypeError(
            f"{cls.__name__}.add_unit must satisfy AddUnitPolicy (spawn, init); "
            f"got {cls.add_unit!r}"
        )

    interval: object = getattr(cls, "structural_interval", 1)
    if not isinstance(interval, int) or isinstance(interval, bool) or interval < 1:
        raise TypeError(
            f"{cls.__name__}.structural_interval must be an int >= 1; got {interval!r}"
        )

    _validate_unit_slots(cls)

    if cls.add_conn is not None:
        grid = isinstance(cls.add_conn, ScoreAddConnPolicy)
        proposed = isinstance(cls.add_conn, ProposeAddConnPolicy)
        if grid == proposed:
            raise TypeError(
                f"{cls.__name__}.add_conn must satisfy exactly one of "
                f"ScoreAddConnPolicy (score) or ProposeAddConnPolicy (propose); "
                f"got {cls.add_conn!r}"
            )
        if isinstance(cls.add_conn, ProposeAddConnPolicy):
            _validate_propose_rule(cls, cls.add_conn)
        else:
            _validate_score_rule(cls, cls.add_conn)
        _validate_growth_knobs(cls, cls.add_conn)

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
            `max_levels` is not an int >= 2, or `unit_policy.prune` or `add_unit` is
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
    if cls.unit_policy.prune is not None and capacity is None:
        raise TypeError(
            f"{cls.__name__}.unit_policy.prune requires a unit_capacity: the `PRUNED` "
            "column that marks pruned units exists only with one"
        )
    if cls.add_unit is not None and capacity is None:
        raise TypeError(
            f"{cls.__name__}.add_unit requires a unit_capacity: children are "
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
        NotImplementedError: if `net` declares `prune_unit` or `add_unit` and
            `sharding` is set.
    """
    if sharding is None:
        return
    if net.unit_policy.prune is not None:
        raise NotImplementedError(
            f"{net.__name__}: unit pruning under sharding is not supported yet"
        )
    if net.add_unit is not None:
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
            f"{cls.__name__}.add_conn.num_proposals was renamed: declare "
            "`proposals_per_proposer` (int >= 1) on the propose rule instead."
        )
    n: object = getattr(ac, "proposals_per_proposer", None)
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise TypeError(
            f"{cls.__name__}.add_conn.proposals_per_proposer must be an "
            f"int >= 1; got {n!r}"
        )
    proposer = getattr(ac, "proposer", "per_unit")
    if not isinstance(proposer, str) or proposer not in _PROPOSE_ARITY:
        raise TypeError(
            f"{cls.__name__}.add_conn.proposer must be one of "
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
            f"{cls.__name__}.add_conn: a {proposer!r} proposer's propose "
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
                f"{cls.__name__}.add_conn.{removed} was removed: {repl} "
                "on the growth rule instead."
            )
    selection = getattr(ac, "selection", "top_k")
    if selection not in _SELECTIONS:
        raise TypeError(
            f"{cls.__name__}.add_conn.selection must be one of "
            f"{_SELECTIONS}; got {selection!r}"
        )
    mnpl: object = getattr(ac, "max_new_per_level", None)
    if selection != "all" or mnpl is not None:
        if not isinstance(mnpl, int) or isinstance(mnpl, bool) or mnpl < 1:
            raise TypeError(
                f"{cls.__name__}.add_conn.max_new_per_level must be an "
                f'int >= 1 (required unless selection = "all"); got {mnpl!r}'
            )
    if selection == "threshold" and not callable(getattr(ac, "threshold", None)):
        raise TypeError(
            f'{cls.__name__}.add_conn: selection = "threshold" requires a '
            "`threshold(g)` method on the rule."
        )
    scope = getattr(ac, "level_cap_scope", "source_level")
    if scope not in _LEVEL_CAP_SCOPES:
        raise TypeError(
            f"{cls.__name__}.add_conn.level_cap_scope must be one of "
            f"{_LEVEL_CAP_SCOPES}; got {scope!r}"
        )
    mnps: object = getattr(ac, "max_new_per_step", None)
    if mnps is not None and (
        not isinstance(mnps, int) or isinstance(mnps, bool) or mnps < 1
    ):
        raise TypeError(
            f"{cls.__name__}.add_conn.max_new_per_step must be an int >= 1 "
            f"or None; got {mnps!r}"
        )
    direction = getattr(ac, "direction", "any")
    if direction not in _DIRECTIONS:
        raise TypeError(
            f"{cls.__name__}.add_conn.direction must be one of "
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
            f"{cls.__name__}.add_conn.trigger must be 'every_step', "
            f"('every', n >= 1), 'on_units_added' or 'when'; got {trigger!r}"
        )
    if trigger == "when" and not callable(getattr(ac, "when", None)):
        raise TypeError(
            f"{cls.__name__}.add_conn: trigger = 'when' requires a "
            "`when(g)` method on the rule."
        )
    on_overflow = getattr(ac, "on_overflow", "flag")
    if on_overflow not in ("flag", "error"):
        raise TypeError(
            f"{cls.__name__}.add_conn.on_overflow must be 'flag' or "
            f"'error'; got {on_overflow!r}"
        )
    for flag in ("dedupe_live", "dedupe_step", "allow_self_loops"):
        v = getattr(ac, flag, False)
        if not isinstance(v, bool):
            raise TypeError(f"{cls.__name__}.add_conn.{flag} must be a bool; got {v!r}")


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
            f"{cls.__name__}.add_conn.candidates must be one of "
            f"{_CANDIDATES}; got {candidates!r}"
        )
    if candidates != "exhaustive":
        size: object = getattr(ac, "shortlist_size", None)
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise TypeError(
                f"{cls.__name__}.add_conn.shortlist_size must be an int >= 1 "
                f"for candidates = {candidates!r}; got {size!r}"
            )
        if not callable(getattr(ac, "importance", None)):
            raise TypeError(
                f"{cls.__name__}.add_conn: candidates = {candidates!r} "
                "requires an `importance(u, i, g)` method on the rule."
            )
