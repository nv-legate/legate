# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
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
from collections.abc import Callable
from inspect import Parameter, Signature
from typing import Any, TypeVar

from ..._lib.data.logical_array cimport LogicalArray
from ..._lib.data.logical_store cimport LogicalStore
from ..._lib.data.physical_array cimport PhysicalArray
from ..._lib.data.physical_store cimport PhysicalStore
from ..._lib.data.scalar cimport Scalar
from ..._lib.operation.task cimport AutoTask
from ..._lib.task.task_context cimport TaskContext

from ...types import _Dtype_from_python_type

from .type cimport (
    InputArray,
    InputStore,
    OutputArray,
    OutputStore,
    ParamList,
    ReductionStore,
)

from .type import UserFunction

from .util cimport RESERVED_ARG_NAMES

_T = TypeVar("_T")
_U = TypeVar("_U")

cdef tuple[type, ...] _BASE_TYPES = (PhysicalStore, PhysicalArray)
cdef tuple[type, ...] _BASE_LOGICAL_TYPES = (LogicalStore, LogicalArray)
cdef tuple[type, ...] _INPUT_TYPES = (InputStore, InputArray)
cdef tuple[type, ...] _OUTPUT_TYPES = (OutputStore, OutputArray)
cdef tuple[type, ...] _OBJECT_TYPES = _INPUT_TYPES + _OUTPUT_TYPES


cdef class VariantInvoker:
    r"""Encapsulate the calling conventions between a user-supplied task
    variant function, and a Legate task."""

    assert (
        RESERVED_ARG_NAMES == {"task_constraints"}
    ), (
        "Must update VariantInvoker __init__ documentation, "
        "as Cython does not allow dynamic __doc__"
    )

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
        positional or keyword arguments, *args and **kwargs are not allowed.
        Default arguments are not yet supported either.

        ``func`` must not contain any arguments named in {"task_constraints"}.
        These are reserved by the implementation, and are specially handled
        at the callsite.
        """
        signature = VariantInvoker._get_signature(func)

        ret_type = signature.return_annotation
        if ret_type is not None and ret_type != Signature.empty:
            raise TypeError(
                "Task must not return values, "
                f"expected 'None' as return-type, found {ret_type}"
            )

        cdef list[str] inputs = []
        cdef list[str] outputs = []
        cdef list[str] reductions = []
        cdef list[str] scalars = []
        cdef str name

        for name, param_descr in signature.parameters.items():
            if name in RESERVED_ARG_NAMES:
                raise TypeError(
                    f'Parameter name "{name}" is not allowed; it is '
                    "reserved by the implementation"
                )

            ty = param_descr.annotation
            if ty is Signature.empty:
                raise TypeError(
                    f"Untyped parameters are not allowed, found {param_descr}"
                )

            if param_descr.kind != Parameter.POSITIONAL_OR_KEYWORD:
                raise NotImplementedError(
                    "'/', '*', '*args', '**kwargs' "
                    "not yet allowed in parameter list"
                )

            if param_descr.default is not Parameter.empty:
                raise NotImplementedError(
                    f"Default values for {ty} not yet supported"
                )

            if issubclass(ty, _INPUT_TYPES):
                inputs.append(name)
            elif issubclass(ty, _OUTPUT_TYPES):
                outputs.append(name)
            elif issubclass(ty, ReductionStore):
                reductions.append(name)
            else:
                if issubclass(ty, _BASE_TYPES):
                    # Is a bare Store/Array an input? an output? who knows!
                    raise NotImplementedError(f"Don't know how to handle {ty}")
                scalars.append(name)

        self._signature = signature
        self._inputs = tuple(inputs)
        self._outputs = tuple(outputs)
        self._reductions = tuple(reductions)
        self._scalars = tuple(scalars)

    @property
    def inputs(self) -> ParamList:
        r"""Return the derived input parameters for a user variant function.

        Returns
        -------
        inputs : ParamList
            The list of paramater names determined to be inputs.
        """
        return self._inputs

    @property
    def outputs(self) -> ParamList:
        r"""Return the derived output parameters for a user variant
        function.

        Returns
        -------
        outputs : ParamList
            The list of paramater names determined to be outputs.
        """
        return self._outputs

    @property
    def reductions(self) -> ParamList:
        r"""Return the derived reduction parameters for a user variant
        function.

        Returns
        -------
        reductions : ParamList
            The list of paramater names determined to be reductions.
        """
        return self._reductions

    @property
    def scalars(self) -> ParamList:
        r"""Return the derived scalar parameters for a user variant
        function.

        Returns
        -------
        scalars : ParamList
            The list of paramater names determined to be scalars.
        """
        return self._scalars

    @property
    def signature(self) -> Signature:
        r"""Return the signature of the user function.

        Returns
        -------
        signature : Signature
            The signature object which describes the user variant function.
        """
        return self._signature

    @staticmethod
    cdef void _handle_param(
        task: AutoTask,
        handled: dict[str, int],
        expected_param: Parameter,
        user_param: Any
    ):
        cdef str param_name = expected_param.name
        # this lookup never fails
        handled[param_name] += 1
        if handled[param_name] > 1:
            raise ValueError(
                f"Got multiple values for argument {param_name}"
            )

        cdef type expected_ty = expected_param.annotation
        # Note issubclass(), expected_ty is the class itself, not
        # an instance of it!
        if issubclass(expected_ty, _BASE_TYPES):
            if not isinstance(user_param, _BASE_LOGICAL_TYPES):
                raise TypeError(
                    f"Argument: '{param_name}' "
                    f"expected one of {_BASE_LOGICAL_TYPES}, "
                    f"got {type(user_param)}"
                )

            if user_param.unbound:
                raise NotImplementedError(
                    "Unbound arrays or stores are not yet supported"
                )

            if issubclass(expected_ty, _INPUT_TYPES):
                # Would have used
                # isinstance(user_param, _BASE_LOGICAL_TYPES)
                # here, but mypy balks:
                #
                # legate/core/task/invoker.py:261:36: error: Argument 1 to
                # "add_input" of "AutoTask" has incompatible type "object";
                # expected "LogicalArray | LogicalStore"  [arg-type]
                #   task.add_input(user_param)
                #                  ^~~~~~~~~~
                assert isinstance(user_param, (LogicalStore, LogicalArray))
                task.add_input(user_param)
            elif issubclass(expected_ty, _OUTPUT_TYPES):
                assert isinstance(user_param, (LogicalStore, LogicalArray))
                task.add_output(user_param)
            elif issubclass(expected_ty, ReductionStore):
                raise NotImplementedError()
                task.add_reduction(user_param, None)  # type: ignore
            else:
                raise NotImplementedError(
                    f"Unsupported parameter type {expected_ty}"
                )
        else:
            if not isinstance(user_param, expected_ty):
                raise TypeError(
                    f"Task expected a value of type {expected_ty} for "
                    f"parameter {param_name}, but got {type(user_param)}"
                )

            task.add_scalar_arg(
                user_param, dtype=_Dtype_from_python_type(type(user_param))
            )

    # Cannot cpdef prepare_call() since: "closures inside cpdef functions not
    # yet supported"
    cpdef void prepare_call(
        self, task: AutoTask, args: tuple[Any, ...], kwargs: dict[str, Any]
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
        params = self.signature.parameters

        if len(params.values()) < len(args):
            raise TypeError(
                f"Task expects {len(params.values())} parameters, "
                f"but {len(args)} were passed"
            )

        cdef str param_name
        cdef dict[str, int] handled = {
            param_name: 0 for param_name in params.keys()
        }

        # Handle positional arguments
        for expected_param, pos_param in zip(params.values(), args):
            VariantInvoker._handle_param(
                task, handled, expected_param, pos_param
            )

        # Handle kwargs
        cdef set unhandled_kwargs = set(kwargs.keys())
        cdef str name

        for name, sig in params.items():
            try:
                param = kwargs[name]
            except KeyError:
                continue

            unhandled_kwargs.remove(name)
            VariantInvoker._handle_param(task, handled, sig, param)

        if unhandled_kwargs:
            raise TypeError(
                "Task does not have keyword argument(s): "
                f"{', '.join(map(str, unhandled_kwargs))}"
            )

        cdef int count
        if missing_params := [
            params[name] for name, count in handled.items() if count == 0
        ]:
            raise TypeError(
                f"missing {len(missing_params)} required argument(s): "
                f"{', '.join(map(str, missing_params))}"
            )

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
            name: str, arg: PhysicalArray
        ) -> PhysicalArray | PhysicalStore:
            cdef type arg_ty = params[name].annotation
            if issubclass(arg_ty, PhysicalArray):
                return arg
            if issubclass(arg_ty, PhysicalStore):
                return arg.data()
            # this is a bug
            raise TypeError(
                f"Unhandled argument type '{arg_ty}' during unpacking, "
                "this is a bug in legate.core!"
            )

        def unpack_scalar(name: str, arg: Scalar) -> object:
            return arg.value()

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
            kw.update(
                {name: unpacker(name, val) for name, val in zip(names, vals)}
            )

        unpack_args(self.inputs, ctx.inputs, maybe_unpack_array)
        unpack_args(self.outputs, ctx.outputs, maybe_unpack_array)
        unpack_args(self.reductions, ctx.reductions, maybe_unpack_array)
        unpack_args(self.scalars, ctx.scalars, unpack_scalar)
        return func(**kw)

    @staticmethod
    cdef object _get_signature(func: Any):
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
        valid : bool
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
