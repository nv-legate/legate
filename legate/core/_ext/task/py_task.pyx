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

from collections.abc import Sequence
from typing import Any

from ..._lib.data.logical_array cimport LogicalArray
from ..._lib.data.logical_store cimport LogicalStore
from ..._lib.mapping.mapping cimport TASK_TARGET_TO_VARIANT_KIND

# import deliberate here, we want the Python enum

from ..._lib.mapping.mapping import TaskTarget

from ..._lib.operation.task cimport AutoTask
from ..._lib.partitioning.constraint cimport Constraint, ConstraintProxy
from ..._lib.runtime.library cimport Library
from ..._lib.runtime.runtime cimport get_legate_runtime
from ..._lib.task.task_context cimport TaskContext
from ..._lib.task.task_info cimport TaskInfo
from .invoker cimport VariantInvoker
from .type cimport VariantKind, VariantList, VariantMapping
from .util cimport RESERVED_ARG_NAMES, validate_variant

from .type import UserFunction


cdef class PyTask:
    r"""A Legate task constructed from a Python callable."""
    def __init__(
        self,
        *,
        func: UserFunction,
        variants: VariantList,
        invoker: VariantInvoker | None = None,
        library: Library | None = None,
        register: bool = True,
    ) -> None:
        r"""Construct a ``PyTask``.

        Parameters
        ----------
        func : UserFunction
            The base user function to invoke in the task.
        variants : VariantList
            The list of variants for which ``func`` is applicable.
        invoker : VariantInvoker, optional
            The invoker used to store the signature and marshall arguments to
            and manage invoking the user variants. Defaults to constructing the
            invoker from ``func``.
        library : Library, optional
            The library context under which to register the new task. Defaults
            to the core context.
        register : bool, True
            Whether to immediately register the task with ``context``. If
            ``False``, the user must manually register the task (via
            ``PyTask.complete_registration()``) before use.
        """
        # Cython has no support for class variables...
        self.UNREGISTERED_ID = -1

        if library is None:
            library = get_legate_runtime().core_library

        if invoker is None:
            invoker = VariantInvoker(func)

        cdef str name
        try:
            name = func.__qualname__
        except AttributeError:
            try:
                name = func.__class__.__qualname__
            except AttributeError:
                name = func.__name__

        self._name = name
        self._invoker = invoker
        self._variants = self._init_variants(func, variants)
        self._task_id = self.UNREGISTERED_ID
        self._library = library
        if register:
            self.complete_registration()

    @property
    def registered(self) -> bool:
        r"""Query whether a ``PyTask`` has completed registration.

        Returns
        -------
        registered : bool
            ``True`` if ``self`` is registered, ``False`` otherwise.
        """
        return self._task_id != self.UNREGISTERED_ID

    @property
    def task_id(self) -> int64_t:
        r"""Return the context-local task ID for this task.

        Returns
        -------
        task_id : int
            The local task ID.

        Raises
        ------
        RuntimeError
            If the task has not completed registration.
        """
        if not self.registered:
            raise RuntimeError(
                "Task must complete registration "
                "(via task.complete_registration()) before receiving a task id"
            )
        return self._task_id

    def prepare_call(self, *args: Any, **kwargs: Any) -> AutoTask:
        r"""Prepare a task instance for execution.

        Parameters
        ----------
        *args : Any, optional
            The positional arguments to the task.
        **kwargs : Any, optional
            The keyword arguments to the task.

        Returns
        -------
        task : AutoTask
            The configured task instance.

        Raises
        ------
        RuntimeError
            If the task has not completed registration before calling this
            method.
        TypeError
            If the type of an argument does not match the expected type
            of the corresponding argument to the task.
        ValueError
            If multiple arguments are given for a single argument. This may
            occur, for example, when a keyword argument overlaps with a
            positional argument.

        Notes
        -----
        It is the user's responsibility to invoke the returned task instance,
        either through `task.execute()` or `get_legate_runtime().submit(task)`.

        The user is not allowed to add any additional inputs, outputs, scalars
        or reductions to `task` after this routine returns.

        See Also
        --------
        legate.core.task.task.PyTask.__call__
        """
        cdef AutoTask task = get_legate_runtime().create_auto_task(
            self._library, self.task_id
        )
        task.throws_exception(RuntimeError)
        self._invoker.prepare_call(task, args, kwargs)
        task.lock()
        return task

    @staticmethod
    cdef Constraint _sanitize_constraint(
        task: AutoTask, constraint: Constraint | ConstraintProxy
    ):
        if isinstance(constraint, Constraint):
            return constraint

        if isinstance(constraint, ConstraintProxy):
            sanitized_args = []
            for arg in constraint.args:
                if isinstance(arg, (LogicalStore, LogicalArray)):
                    if isinstance(arg, LogicalStore):
                        arg = LogicalArray.from_store(arg)
                    arg = task.find_or_declare_partition(arg)
                sanitized_args.append(arg)
            return constraint.func(*sanitized_args)

        raise TypeError(
            f"Expected a constraint but got {type(constraint)}"
        )

    assert (
        "task_constraints" in RESERVED_ARG_NAMES
    ), "task_constraints no longer reserved? Has PyTask.__call__ been updated?"

    def __call__(
        self,
        *args: Any,
        task_constraints: Sequence[Constraint | ConstraintProxy] | None = None,
        **kwargs: Any,
    ) -> None:
        r"""Invoke the task.

        Parameters
        ----------
        *args : Any, optional
            The positional arguments to the task.
        task_constraints : Sequence[Constraint | ConstraintProxy], optional
            The set of constraints to impose on the task arguments.
        **kwargs : Any, optional
            The keyword arguments to the task.

        Notes
        -----
        This method is equivalent to the following:

        ::

            task_inst = task.prepare_call(*args, **kwargs)
            if task_constraints is not None:
                # add constraints to task_inst
                ...
            task_inst.execute()


        As a result, it has the same exception and usage profile as
        `PyTask.prepare_call`.

        See Also
        --------
        legate.core.task.task.PyTask.prepare_call
        """
        cdef AutoTask task = self.prepare_call(*args, **kwargs)
        if task_constraints is not None:
            for cnst in task_constraints:
                task.add_constraint(PyTask._sanitize_constraint(task, cnst))
        task.execute()

    cpdef int64_t complete_registration(self):
        r"""Complete registration for a task.

        Returns
        -------
        task_id : int
            The (context-local) task ID for the task.

        Raises
        ------
        ValueError
            If the task has no registered variants.

        Notes
        -----
        It is safe to call this method on an already registered task (it does
        nothing).
        """
        if self.registered:
            return self._task_id

        cdef dict proc_kind_to_variant = {
            TaskTarget.GPU: self._gpu_variant,
            TaskTarget.CPU: self._cpu_variant,
            TaskTarget.OMP: self.omp_variant,
        }

        cdef list variants = [
            (
                TASK_TARGET_TO_VARIANT_KIND[proc_kind],
                proc_kind_to_variant[proc_kind],
            )
            for proc_kind, fn in self._variants.items()
            if fn is not None
        ]
        if not variants:
            raise ValueError("Task has no registered variants")

        cdef int64_t task_id = self._library.get_new_task_id()
        cdef TaskInfo task_info = TaskInfo.from_variants(
            task_id, self._name, variants
        )
        self._library.register_task(task_info)
        self._task_id = task_id
        return task_id

    cdef void _update_variant(self, func: UserFunction, variant: TaskTarget):
        if self.registered:
            raise RuntimeError(
                f"Task (id: {self._task_id}) has already completed "
                "registration and cannot update its variants"
            )
        validate_variant(variant.name.casefold())
        self._invoker.validate_signature(func)
        self._variants[variant] = func

    cpdef void cpu_variant(self, func: UserFunction):
        r"""Register a CPU variant for this task

        Parameters
        ----------
        func : UserFunction
            The new CPU variant function to call for all CPU executions.

        Raises
        ------
        RuntimeError
            If the task has already completed registration.

        Notes
        -----
        Calling this method on a task with an existing CPU variant replaces
        the old variant with ``func``. Therefore, this method may be used to
        update variants as well as add new ones.
        """
        self._update_variant(func, TaskTarget.CPU)

    cpdef void gpu_variant(self, func: UserFunction):
        r"""Register a GPU variant for this task

        Parameters
        ----------
        func : UserFunction
            The new GPU variant function to call for all GPU executions.

        Raises
        ------
        RuntimeError
            If the task has already completed registration.

        Notes
        -----
        Calling this method on a task with an existing GPU variant replaces
        the old variant with ``func``. Therefore, this method may be used to
        update variants as well as add new ones.
        """
        self._update_variant(func, TaskTarget.GPU)

    cpdef void omp_variant(self, func: UserFunction):
        r"""Register an OpenMP variant for this task

        Parameters
        ----------
        func : UserFunction
            The new OpenMP variant function to call for all OpenMP executions.

        Raises
        ------
        RuntimeError
            If the task has already completed registration.

        Notes
        -----
        Calling this method on a task with an existing OpenMP variant replaces
        the old variant with ``func``. Therefore, this method may be used to
        update variants as well as add new ones.
        """
        self._update_variant(func, TaskTarget.OMP)

    cdef VariantMapping _init_variants(
        self,
        func: UserFunction,
        variants: VariantList
    ):
        cdef VariantKind v
        for v in variants:
            validate_variant(v)
        self._invoker.validate_signature(func)
        return {
            TaskTarget.CPU: func if "cpu" in variants else None,
            TaskTarget.GPU: func if "gpu" in variants else None,
            TaskTarget.OMP: func if "omp" in variants else None,
        }

    cdef void _invoke_variant(self, ctx: TaskContext, kind: TaskTarget):
        assert (
            variant := self._variants[kind]
        ) is not None, f"Task has no variant for kind: {kind}"
        self._invoker(ctx, variant)

    cdef void _cpu_variant(self, TaskContext ctx):
        self._invoke_variant(ctx, TaskTarget.CPU)

    cdef void _gpu_variant(self, TaskContext ctx):
        self._invoke_variant(ctx, TaskTarget.GPU)

    cdef void _omp_variant(self, TaskContext ctx):
        self._invoke_variant(ctx, TaskTarget.OMP)
