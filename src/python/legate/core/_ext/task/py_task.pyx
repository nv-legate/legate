# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from ..._lib.operation.task cimport AutoTask
from ..._lib.runtime.library cimport Library
from ..._lib.runtime.runtime cimport get_legate_runtime
from ..._lib.task.task_context cimport TaskContext
from ..._lib.task.task_info cimport TaskInfo
from ..._lib.task.task_signature cimport _TaskSignature
from ..._lib.utilities.typedefs cimport VariantCode, _LocalTaskID
from ..._lib.utilities.typedefs import VariantCode as PyVariantCode
from .invoker cimport VariantInvoker
from .type cimport VariantList, VariantMapping
from .util cimport validate_variant, _get_callable_name

from .type import UserFunction


cdef class PyTask:
    r"""A Legate task constructed from a Python callable."""
    def __init__(
        self,
        *,
        func: UserFunction,
        variants: VariantList,
        constraints: Sequence[ConstraintProxy] | None = None,
        throws_exception: bool = False,
        has_side_effect: bool = False,
        invoker: VariantInvoker | None = None,
        library: Library | None = None,
        register: bool = True,
    ) -> None:
        r"""Construct a ``PyTask``.

        Parameters
        ----------
        func
            The base user function to invoke in the task.
        variants
            The list of variants for which ``func`` is applicable.
        constraints
            The list of constraints which are to be applied to the arguments of
            ``func``, if any. Defaults to no constraints.
        throws_exception
            ``True`` if any variants of ``func`` throws an exception, ``False``
            otherwise.
        has_side_effect : bool, False
            Whether the task has any global side-effects. See
            ``AutoTask.set_side_effect()`` for further information.
        invoker
            The invoker used to store the signature and marshall arguments to
            and manage invoking the user variants. Defaults to constructing the
            invoker from ``func``.
        library
            The library context under which to register the new task. Defaults
            to the core context.
        register
            Whether to immediately register the task with ``library``. If
            ``False``, the user must manually register the task (via
            ``PyTask.complete_registration()``) before use.
        """
        # Cython has no support for class variables...
        self.UNREGISTERED_ID = <_LocalTaskID>-1

        if library is None:
            library = get_legate_runtime().core_library

        if invoker is None:
            invoker = VariantInvoker(func, constraints=constraints)

        self._name = _get_callable_name(func)
        self._invoker = invoker
        self._variants = self._init_variants(func, variants)
        self._task_id = self.UNREGISTERED_ID
        self._library = library
        self._throws = throws_exception
        self._has_side_effect = has_side_effect
        if register:
            self.complete_registration()

    @property
    def registered(self) -> bool:
        r"""Query whether a ``PyTask`` has completed registration.

        :return: ``True`` if ``self`` is registered, ``False`` otherwise.
        :rtype: bool
        """
        return self._task_id != self.UNREGISTERED_ID

    @property
    def task_id(self) -> _LocalTaskID:
        r"""Return the context-local task ID for this task.

        :return: The local task ID of the task.
        :rtype: LocalTaskID

        :raises RuntimeError: If the task has not completed registration.
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
        AutoTask
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
        legate.task.task.PyTask.__call__
        """
        cdef AutoTask task = get_legate_runtime().create_auto_task(
            self._library, self.task_id
        )
        task._handle.throws_exception(self._throws)
        task.set_side_effect(self._has_side_effect)
        self._invoker.prepare_call(task, args, kwargs)
        task.lock()
        return task

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        r"""Invoke the task.

        Parameters
        ----------
        *args : Any, optional
            The positional arguments to the task.
        **kwargs : Any, optional
            The keyword arguments to the task.

        Notes
        -----
        This method is equivalent to the following:

        ::

            task.prepare_call(*args, **kwargs).execute()


        As a result, it has the same exception and usage profile as
        `PyTask.prepare_call`.

        See Also
        --------
        legate.task.task.PyTask.prepare_call
        """
        self.prepare_call(*args, **kwargs).execute()

    cpdef _LocalTaskID complete_registration(self):
        r"""Complete registration for a task.

        Returns
        -------
        LocalTaskID
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
            VariantCode.CPU: self._cpu_variant,
            VariantCode.GPU: self._gpu_variant,
            VariantCode.OMP: self._omp_variant,
        }

        cdef VariantCode v

        cdef list variants = [
            (v, proc_kind_to_variant[v])
            for v, fn in self._variants.items()
            if fn is not None
        ]
        if not variants:
            raise ValueError("Task has no registered variants")

        cdef _LocalTaskID task_id = self._library.get_new_task_id()
        cdef _TaskSignature signature = self._invoker.prepare_task_signature()
        cdef TaskInfo task_info = TaskInfo.from_variants_signature(
            task_id,
            self._name,
            variants,
            &signature
        )
        self._library.register_task(task_info)
        self._task_id = task_id
        return task_id

    cdef void _update_variant(self, func: UserFunction, VariantCode variant):
        if self.registered:
            raise RuntimeError(
                f"Task (id: {self._task_id}) has already completed "
                "registration and cannot update its variants"
            )

        validate_variant(variant)
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
        self._update_variant(func, VariantCode.CPU)

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
        self._update_variant(func, VariantCode.GPU)

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
        self._update_variant(func, VariantCode.OMP)

    cdef VariantMapping _init_variants(
        self,
        func: UserFunction,
        variants: VariantList
    ):
        cdef VariantCode v
        for v in variants:
            validate_variant(v)
        self._invoker.validate_signature(func)

        return {v: func if v in variants else None for v in PyVariantCode}

    cdef void _invoke_variant(self, TaskContext ctx, VariantCode variant):
        variant_impl = self._variants[variant]
        assert variant_impl is not None, (
            f"Task has no variant for kind: {variant}"
        )
        self._invoker(ctx, variant_impl)

    cdef void _cpu_variant(self, TaskContext ctx):
        self._invoke_variant(ctx, VariantCode.CPU)

    cdef void _gpu_variant(self, TaskContext ctx):
        self._invoke_variant(ctx, VariantCode.GPU)

    cdef void _omp_variant(self, TaskContext ctx):
        self._invoke_variant(ctx, VariantCode.OMP)
