# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from __future__ import annotations

from libcpp cimport bool

import inspect
import warnings
from collections.abc import Callable
from inspect import Parameter, Signature
from pickle import (
    HIGHEST_PROTOCOL as PKL_HIGHEST_PROTOCOL,
    dumps as pkl_dumps,
    loads as pkl_loads,
)
from pickletools import optimize as pklt_optimize
from types import NoneType, UnionType
from typing import (
    Any,
    TypeVar,
    _UnionGenericAlias,
    get_args as typing_get_args,
    get_origin as typing_get_origin,
)

from ..._lib.data.logical_array cimport LogicalArray
from ..._lib.data.logical_store cimport LogicalStore
from ..._lib.data.physical_array cimport PhysicalArray
from ..._lib.data.physical_store cimport PhysicalStore
from ..._lib.data.scalar cimport Scalar
from ..._lib.operation.task cimport AutoTask
from ..._lib.runtime.runtime cimport get_legate_runtime
from ..._lib.task.task_context cimport TaskContext
from ..._lib.type.type_info cimport Type, TypeCode, binary_type

from ...data_interface import (
    MAX_DATA_INTERFACE_VERSION,
    MIN_DATA_INTERFACE_VERSION,
    LegateDataInterface,
)

from .type cimport (
    ConstraintSet,
    InputArray,
    InputStore,
    OutputArray,
    OutputStore,
    ParamList,
)

from .type import ReductionArray, ReductionStore, UserFunction


cdef object _T = TypeVar("_T")
cdef object _U = TypeVar("_U")

cdef tuple[type, ...] _BASE_PHYSICAL_TYPES = (PhysicalStore, PhysicalArray)
cdef tuple[type, ...] _BASE_LOGICAL_TYPES = (LogicalStore, LogicalArray)
cdef tuple[type, ...] _BASE_TYPES = _BASE_PHYSICAL_TYPES + _BASE_LOGICAL_TYPES
cdef tuple[type, ...] _INPUT_TYPES = (InputStore, InputArray)
cdef tuple[type, ...] _OUTPUT_TYPES = (OutputStore, OutputArray)
cdef tuple[type, ...] _REDUCTION_TYPES = (ReductionStore, ReductionArray)


cdef try_unpack_simple_store(arg: LegateDataInterface, name: str):
    iface = arg.__legate_data_interface__

    if "version" not in iface:
        raise TypeError(
            f"Argument: {name!r} Legate data interface missing a version "
            "number"
        )

    v = iface["version"]

    if not isinstance(v, int):
        raise TypeError(
            f"Argument: {name!r} Legate data interface version expected an "
            f"integer, got {v!r}"
        )

    if v < MIN_DATA_INTERFACE_VERSION:
        raise TypeError(
            f"Argument: {name!r} Legate data interface version {v} is below "
            f"{MIN_DATA_INTERFACE_VERSION=}"
        )

    if v > MAX_DATA_INTERFACE_VERSION:
        raise NotImplementedError(
            f"Argument: {name!r} Unsupported Legate data interface version {v}"
        )

    data = iface["data"]

    it = iter(data)

    try:
        field = next(it)
    except StopIteration:
        raise TypeError(f"Argument: {name!r} Legate data object has no fields")

    try:
        next(it)
    except StopIteration:
        pass
    else:
        raise NotImplementedError(
            f"Argument: {name!r} Legate data interface objects with more than "
            "one store are unsupported"
        )

    if field.nullable:
        raise NotImplementedError(
            f"Argument: {name!r} Legate data interface objects with nullable "
            "fields are unsupported"
        )

    column = data[field]
    if column.nullable:
        raise NotImplementedError(
            f"Argument: {name!r} Legate data interface objects with nullable "
            "stores are unsupported"
        )

    return column.data


cdef inline type _unpack_generic_type(object annotation):
    origin_type = typing_get_origin(annotation)

    if origin_type is None:
        # typing.get_origin() "returns None if the type is not supported". In
        # practice this just means that the annotation was not a generic. In
        # this case we hope that it's some kind of type, otherwise Cython will
        # balk when it tries to convert.
        assert isinstance(annotation, type), (
            f"Unhandled type annotation: {annotation}, expected this to be a "
            f"type, got {type(annotation)} instead"
        )
        return annotation
    return origin_type

cdef inline type _unpack_union_type(object annotation):
    cdef tuple[type, ...] sub_types = typing_get_args(annotation)

    if len(sub_types) != 2:
        raise NotImplementedError(
            "Arbitrary union types not yet supported. Union types may "
            "only be 'SomeType | None' (order doesn't matter), "
            "'Union[SomeType, None]' (order doesn't matter), or "
            f"'Optional[SomeType]'. Found '{annotation}'."
        )

    cdef type ty
    cdef type ret = None

    # Want to find the first (read: the only) non-None type in the list.
    for ty in sub_types:
        if not issubclass(ty, NoneType):
            if ret is not None:
                # Been here before, means the union type has multiple non-None
                # types
                raise NotImplementedError(
                    "Arbitrary union types not yet supported. Union types may "
                    "only be 'SomeType | None' (order doesn't matter), "
                    "'Union[SomeType, None]' (order doesn't matter), or "
                    f"'Optional[SomeType]'. Found '{annotation}' "
                    f"(found '{ty}', previous non-None type '{ret}')."
                )
            # Check above would break if ty is None
            assert ty is not None
            ret = ty

    return ret


cdef void _assert_union_types_are_what_we_expect():
    import builtins
    from typing import Optional, Union

    # Use builtins module here to make triply sure that Cython isn't
    # transforming any of these types, and that they are pure python
    # types. Cannot use builtins.None because None is a keyword, so we have to
    # getattr it.
    x = builtins.int | getattr(builtins, "None")
    assert isinstance(x, UnionType)
    x = Optional[builtins.int]
    assert isinstance(x, _UnionGenericAlias)
    x = Union[builtins.int, getattr(builtins, "None")]
    assert isinstance(x, _UnionGenericAlias)


_assert_union_types_are_what_we_expect()

cdef inline type _unpack_type(object annotation):
    if isinstance(annotation, (UnionType, _UnionGenericAlias)):
        # _UnionGenericAlias is needed to catch all 3 of the "union" variants:
        #
        # 1. 'x | y' -> UnionType
        # 2. 'Optional[x]' -> _UnionGenericAlias
        # 3. 'Union[x, y]' -> _UnionGenericAlias
        return _unpack_union_type(annotation)
    return _unpack_generic_type(annotation)


cdef tuple[
    tuple[str], tuple[str], tuple[str], tuple[str]
] _parse_signature(object signature):
    cdef list[str] inputs = []
    cdef list[str] outputs = []
    cdef list[str] reductions = []
    cdef list[str] scalars = []
    cdef str name
    cdef type ty
    cdef int num_redops

    for name, param_descr in signature.parameters.items():
        annotation = param_descr.annotation
        if annotation is Signature.empty:
            raise TypeError(
                f"Untyped parameters are not allowed, found {param_descr}"
            )

        if param_descr.kind != Parameter.POSITIONAL_OR_KEYWORD:
            raise NotImplementedError(
                "'/', '*', '*args', '**kwargs' "
                "not yet allowed in parameter list"
            )

        default_var = param_descr.default
        if (default_var is not Parameter.empty) and (
            isinstance(default_var, _BASE_TYPES)
        ):
            raise NotImplementedError(
                f"Default values for {annotation} not yet supported"
            )

        ty = _unpack_type(annotation)

        if issubclass(ty, _INPUT_TYPES):
            inputs.append(name)
        elif issubclass(ty, _OUTPUT_TYPES):
            outputs.append(name)
        elif issubclass(ty, _REDUCTION_TYPES):
            # Reduction stores, which are typing._GenericAlias (not a
            # type!)
            if (num_redops := len(typing_get_args(annotation))) != 1:
                raise TypeError(
                    f"Type hint '{annotation}' has an invalid number of "
                    f"reduction operators ({num_redops}), expected 1. "
                    f"For example: '{name}: {annotation.__name__}[ADD]'"
                )
            reductions.append(name)
        elif issubclass(ty, _BASE_TYPES):
            # Is a bare Store/Array an input? an output? who knows!
            raise TypeError(
                f"Type hint '{annotation}' is invalid, because it is "
                "impossible to deduce intent from it. Must use either "
                "Input/Output/Reduction variant"
            )
        else:
            scalars.append(name)

    return (
        tuple(inputs),
        tuple(outputs),
        tuple(reductions),
        tuple(scalars),
    )

cdef inline bytes _serialize_object(object value):
    return pklt_optimize(pkl_dumps(value, protocol=PKL_HIGHEST_PROTOCOL))

cdef bytes LEGATE_PICKLE_HEADER = b"__legate_pickled_arg__"

cdef class VariantInvoker:
    r"""Encapsulate the calling conventions between a user-supplied task
    variant function, and a Legate task."""

    def __init__(self, func: UserFunction) -> None:
        r"""Construct a ``VariantInvoker``

        Parameters
        ----------
        func : UserFunction
            The user function which is to be invoked.

        Raises
        ------
        TypeError
            If ``func`` has a non-conforming signature.

        Notes
        -----
        All parameters to ``func`` which are neither inputs, outputs, or
        reductions, are automatically considered to be scalars.

        All user functions must return exactly ``None``, and all arguments
        must be fully type-hinted. Furthermore, all arguments must be
        positional or keyword arguments, ``*args`` and ``**kwargs`` are not
        allowed.

        Default arguments are not yet supported either.
        """
        signature = VariantInvoker._get_signature(func)

        ret_type = signature.return_annotation
        if ret_type is not None and ret_type != Signature.empty:
            raise TypeError(
                "Task must not return values, "
                f"expected 'None' as return-type, found {ret_type}"
            )

        self._signature = signature
        (
            self._inputs,
            self._outputs,
            self._reductions,
            self._scalars
        ) = _parse_signature(signature)

    @property
    def inputs(self) -> ParamList:
        r"""
        Return the derived input parameters for a user variant function.

        :returns: The list of paramater names determined to be inputs.
        :rtype: ParamList
        """
        return self._inputs

    @property
    def outputs(self) -> ParamList:
        r"""Return the derived output parameters for a user variant function.

        :returns: The list of paramater names determined to be outputs.
        :rtype: ParamList
        """
        return self._outputs

    @property
    def reductions(self) -> ParamList:
        r"""
        Return the derived reduction parameters for a user variant function.

        :returns: The list of paramater names determined to be reductions.
        :rtype: ParamList
        """
        return self._reductions

    @property
    def scalars(self) -> ParamList:
        r"""Return the derived scalar parameters for a user variant function.

        :returns: The list of paramater names determined to be scalars.
        :rtype: ParamList
        """
        return self._scalars

    @property
    def signature(self) -> Signature:
        r"""Return the signature of the user function.

        :returns: The signature object which describes the user variant
                  function.
        :rtype: inspect.Signature
        """
        return self._signature

    @staticmethod
    cdef object _handle_param(
        task: AutoTask,
        expected_param: Parameter,
        user_param: Any
    ):
        annotation = expected_param.annotation

        cdef type expected_ty = _unpack_type(annotation)

        # Note issubclass(), expected_ty is the class itself, not
        # an instance of it!
        if issubclass(expected_ty, _BASE_PHYSICAL_TYPES):
            if user_param is None:
                # Special case for None. We cannot just pass None to the
                # add_input/output functions below, they expect a valid store
                # object. Furthermore, we also need something to fill the
                # "hole" that this store would occupy in the list of
                # inputs/outputs that we get when inside the task body.
                #
                # So we create a dummy store that contains a marker value which
                # we can check for in the un-marshalling step in __call__().
                #
                # Another alternative would be to create an empty, unbound
                # store (which is preferred as it occupies the least space and
                # is cheapest to construct), and attach some kind of "this is a
                # dummy object" metadata tag to it. But at the time of writing,
                # metadata on stores does not yet exist.
                user_param = get_legate_runtime().create_store_from_scalar(
                    Scalar(None)
                )

            # Special case for "simple" legate data interface objects, e.g
            # cuPyNumeric arrays, that have only a single, non-nullable logical
            # store
            if hasattr(user_param, "__legate_data_interface__"):
                user_param = try_unpack_simple_store(
                    user_param,
                    expected_param.name
                )

            if not isinstance(user_param, _BASE_LOGICAL_TYPES):
                raise TypeError(
                    f"Argument: '{expected_param.name}' "
                    f"expected one of {_BASE_LOGICAL_TYPES}, "
                    f"got {type(user_param)}"
                )

            if user_param.unbound:
                raise NotImplementedError(
                    "Unbound arrays or stores are not yet supported"
                )

            if issubclass(expected_ty, _INPUT_TYPES):
                task.add_input(user_param)
            elif issubclass(expected_ty, _OUTPUT_TYPES):
                task.add_output(user_param)
            elif issubclass(expected_ty, _REDUCTION_TYPES):
                task.add_reduction(user_param, typing_get_args(annotation)[0])
            else:
                raise NotImplementedError(
                    f"Unsupported parameter type {expected_ty}"
                )
        # Must do this elif _after_ we check for physical types above. The type
        # hint says "InputArray" (A.K.A. PhysicalArray), but the user will be
        # passing in LogicalArray.
        elif not isinstance(user_param, expected_ty):
            raise TypeError(
                f"Task expected a value of type {expected_ty} for "
                f"parameter {expected_param.name}, but got {type(user_param)}"
            )
        elif issubclass(expected_ty, Scalar):
            task.add_scalar_arg(user_param)
        else:
            try:
                dtype = Type.from_py_object(user_param)
            except NotImplementedError:
                warnings.warn(
                    f"Argument type: {type(user_param)} not natively "
                    "supported by type inference, falling back to pickling "
                    "(which may incur a slight performance penalty). Consider "
                    "opening a bug report at "
                    "https://github.com/nv-legate/legate.core."
                )
                user_param = (
                    LEGATE_PICKLE_HEADER + _serialize_object(user_param)
                )
                dtype = binary_type(len(user_param))
            task.add_scalar_arg(user_param, dtype=dtype)
        return user_param

    cdef ParamMapping _prepare_params(
        self, AutoTask task, tuple[Any, ...] args, dict[str, Any] kwargs
    ):
        signature = self.signature

        bound_params = signature.bind(*args, **kwargs)
        bound_params.apply_defaults()

        cdef ParamMapping param_mapping = bound_params.arguments
        cdef str name
        # Traversal of the arguments using params is deliberate. We need to
        # keep the relative ordering of the inputs, outputs, etc in the same
        # order in which they were declared in the signature.
        for name, sig in signature.parameters.items():
            param_mapping[name] = VariantInvoker._handle_param(
                task, sig, param_mapping[name]
            )
        return param_mapping

    @staticmethod
    cdef void _prepare_constraints(
        AutoTask task, ParamMapping param_mapping, ConstraintSet constraints
    ):
        r"""Handle any constraints imposed on the task

        Parameters
        ----------
        task : AutoTask
            The task to which to apply the constraints.
        param_mapping : ParamMapping
            The mapping of parameter names to their values.
        constraints : ConstraintSet
            The set of constraint proxies.
        """
        cdef ConstraintProxy constraint
        cdef list sanitized_args

        for constraint in constraints:
            sanitized_args = []
            for arg in constraint.args:
                if isinstance(arg, str):
                    arg_value = param_mapping[arg]
                    if isinstance(arg_value, LogicalStore):
                        arg_value = LogicalArray.from_store(arg_value)
                    else:
                        assert isinstance(arg_value, LogicalArray), (
                            "Parameter constraint argument of unexpected type "
                            f"{type(arg_value)}. Expected LogicalArray or "
                            "LogicalStore."
                        )
                    arg_value = task.find_or_declare_partition(arg_value)
                else:
                    assert arg is not None
                    arg_value = arg
                sanitized_args.append(arg_value)
            task.add_constraint(constraint.func(*sanitized_args))

    cpdef void prepare_call(
        self,
        AutoTask task,
        tuple[Any, ...] args,
        dict[str, Any] kwargs,
        # This should be constraints: ConstraintSet | None, but then Cython
        # barfs with "Signature not compatible with previous declaration", even
        # if you change the .pxd to match. So here we are, lying to Cython.
        ConstraintSet constraints = None
    ):
        r"""Prepare a list of arguments for task call.

        Parameters
        ----------
        task : AutoTask
            The task to prepare the arguments for.
        args : tuple, optional
            The set of positional arguments for the task.
        kwargs : dict, optional
            The set of keyword arguments for the task.

        Raises
        ------
        ValueError
            If multiple arguments are given for a single argument. This may
            occur, for example, when a keyword argument overlaps with a
            positional argument.
        TypeError
            If the type of an argument does not match the expected type
            of the corresponding argument to the task, or if there are missing
            required parameters.
        NameError
            If a keyword argument does not exist in the function's signature.

        Notes
        -----
        ``args`` and ``kwargs`` are not the usual expanded ``tuple`` and
        ``dict``. Instead, they correspond to a literal ``tuple`` and ``dict``
        respectively. That is::

            # Incorrect
            invoker.prepare_call(
                task,
                a, b, c,
                foo="bar", baz="bop"
            )

            # Correct
            invoker.prepare_call(
                task,
                (a, b, c),
                {"foo" : "bar", "baz" : "bop"}
            )
        """
        cdef ParamMapping param_mapping = self._prepare_params(
            task, args, kwargs
        )
        if constraints is None:
            constraints = tuple()
        VariantInvoker._prepare_constraints(task, param_mapping, constraints)

    def __call__(self, ctx: TaskContext, func: UserFunction) -> None:
        r"""Invoke the given function by adapting a TaskContext to the
        parameters for the function.

        Parameters
        ----------
        ctx : TaskContext
            The Legate ``TaskContext`` which describes the task and holds
            the arguments for ``func``.
        func : UserFunction
            The resulting Python callable to invoke.

        Notes
        -----
        Generally the user should not call this method themselves, it is
        invoked as part of the Python task calling sequence.

        Raises
        ------
        ValueError
            If the signature of ``func`` does not match the configured
            signature of this ``VariantInvoker``.
        """
        cdef dict[str, Any] kw = {}

        params = self.signature.parameters

        def maybe_unpack_array(
            default_val: Any, arg_ty: type, arg: PhysicalArray,
        ) -> PhysicalArray | PhysicalStore | None:
            cdef object ret

            if issubclass(arg_ty, PhysicalArray):
                ret = arg
            elif issubclass(arg_ty, PhysicalStore):
                ret = arg.data()
            else:
                # this is a bug
                raise TypeError(
                    f"Unhandled argument type '{arg_ty}' during unpacking, "
                    "this is a bug in legate!"
                )

            if default_val is Parameter.empty:
                return ret

            cdef Type ret_ty = ret.type

            assert default_val is None
            if ret_ty.code == TypeCode.NIL:
                return None
            return ret

        def unpack_scalar(
            default_val: Any, arg_ty: type, arg: Scalar
        ) -> object | Scalar:
            if issubclass(arg_ty, Scalar):
                return arg
            val = arg.value()
            if (
                isinstance(val, memoryview)
                and val[:len(LEGATE_PICKLE_HEADER)] == LEGATE_PICKLE_HEADER
            ):
                # we pickled the object, unpickle it transparently
                return pkl_loads(val[len(LEGATE_PICKLE_HEADER):])
            if isinstance(val, arg_ty):
                return val
            return arg_ty(val)

        def unpack_args(
            names: tuple[str, ...],
            vals: tuple[_T, ...],
            unpacker: Callable[[str, _T], _U],
        ) -> None:
            if len(names) != len(vals):
                raise ValueError(
                    f"Wrong number of given arguments ({len(vals)}), "
                    f"expected {len(names)}"
                )
            cdef str name
            cdef type arg_ty

            for name, val in zip(names, vals):
                sig = params[name]
                arg_ty = _unpack_type(sig.annotation)
                kw[name] = unpacker(sig.default, arg_ty, val)

        unpack_args(self.inputs, ctx.inputs, maybe_unpack_array)
        unpack_args(self.outputs, ctx.outputs, maybe_unpack_array)
        unpack_args(self.reductions, ctx.reductions, maybe_unpack_array)
        unpack_args(self.scalars, ctx.scalars, unpack_scalar)
        return func(**kw)

    @staticmethod
    cdef object _get_signature(object func):
        return inspect.signature(func, eval_str=True)

    cpdef bool valid_signature(self, func: UserFunction):
        r"""Whether the given function's signature  matches the configured
        function signature.

        Parameters
        ----------
        func : UserFunction
            The Python callable whose signature should be validated.

        Returns
        -------
        bool
            ``True`` if the signature of ``func`` matches this
            ``VariantInvoker``s signature, ``False`` otherwise.
        """
        return VariantInvoker._get_signature(func) == self.signature

    cpdef void validate_signature(self, func: UserFunction):
        r"""Ensure a callable's signature matches the configured signature.

        Paramters
        ---------
        func : UserFunction
            The Python callable whose signature should be validated.

        Raises
        ------
        ValueError
            If the signature of ``func`` differs from the configured signature
            of this ``VariantInvoker``.

        Notes
        -----
        This method is a 'raising' version of
        ``VariantInvoker.valid_signature()``, that is::

            is_valid = invoker.valid_signature(func)
            # is equivalent to
            try:
                invoker.validate_signature(func)
            except ValueError:
                is_valid = False
            else:
                is_valid = True
        """
        if not self.valid_signature(func):
            raise ValueError(
                f"Invalid signature for {func}, expected {self.signature}, "
                f"got {VariantInvoker._get_signature(func)}"
            )
