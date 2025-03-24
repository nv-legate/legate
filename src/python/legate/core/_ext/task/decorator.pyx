# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from libcpp cimport bool

from collections.abc import Callable, Sequence

from ..._lib.partitioning.constraint cimport DeferredConstraint
from ..._lib.utilities.typedefs cimport VariantCode

# imports are deliberate here, we want the python objects. Technically not true
# for PyTask, but cimporting it results in:
#
# cmake-build/legate/core/_ext/task/decorator.cxx:4025:17: error: incompatible
# pointer types assigning to 'PyObject *' (aka '_object *') from 'struct
# __pyx_obj_6legate_4core_4_ext_4task_7py_task_PyTask *'
#         __pyx_t_2 =
#         __pyx_pf_6legate_4core_4_ext_4task_9decorator_4task_decorator(__pyx_v_decorator,
#         __pyx_v_func);
#
# So I guess we import it.

from .py_task import PyTask
from .type import UserFunction
from .util import dynamic_docstring


cdef tuple[VariantCode, ...] DEFAULT_VARIANT_LIST = (VariantCode.CPU,)


@dynamic_docstring(DEFAULT_VARIANT_LIST=DEFAULT_VARIANT_LIST)
def task(
    func: UserFunction | None = None,
    *,
    variants: tuple[VariantCode, ...] = DEFAULT_VARIANT_LIST,
    constraints: Sequence[DeferredConstraint] | None = None,
    options: TaskConfig | VariantOptions | None = None,
    register: bool = True,
) -> Callable[[UserFunction], PyTask] | PyTask:
    r"""Convert a Python function to a Legate task.

    The user may pass either a ``TaskConfig`` or ``VariantOptions`` to select
    the task options. If the former, the user can select the task ID of the
    task themselves, while additionally setting variant options via the
    ``TaskConfig.variant_options`` property. If the latter, then a new, unique
    task ID will be automatically generated.

    Deferring registration is used to add additional variants to the task that
    have a different body. However, all variants must have identical
    signatures. The user must manually call ``PyTask.complete_registration`` to
    finish registering the task.

    Parameters
    ----------
    func : UserFunction
        The base user function to invoke in the task.
    variants : tuple[VariantCode, ...], optional
        The list of variants for which ``func`` is applicable. Defaults
        to ``{DEFAULT_VARIANT_LIST}``.
    constraints : Sequence[DeferredConstraint], optional
        The list of constraints which are to be applied to the arguments of
        ``func``, if any. Defaults to no constraints.
    options : TaskConfig | VariantOptions, optional
        Either a ``TaskConfig`` or ``VariantOptions`` describing the task
        configuration.
    register : bool, True
        Whether to immediately complete registration of the task. Deferring
        registration is used to add additional variants to the task that have
        a different body. However, all variants must have identical signatures.
        The user must manually call ``PyTask.complete_registration`` to finish
        registering the task.

    Returns
    -------
    PyTask
        The task object.

    Example
    -------
    ::

        from legate.core import broadcast, align, VariantCode, VariantOptions
        from legate.core.task import task, InputArray, OutputArray

        @task
        def my_basic_task(
            x: InputArray,
            y: OutputArray,
            z: tuple[int, ...] = (1, 2, 3)
         ) -> None:
            ...

        @task(
            variants=(VariantCode.CPU, VariantCode.GPU),
            constraints=(align("x", "y"), broadcast("x")),
            options=VariantOptions(may_throw_exception=True)
        )
        def my_task_with_options(
            x: InputArray,
            y: OutputArray,
            z: tuple[int, ...] = (1, 2, 3)
        ) -> None:
            raise RuntimeError("Exceptional!")


    See Also
    --------
    legate.core.task.task.PyTask.__init__
    """

    def decorator(f: UserFunction) -> PyTask:
        return PyTask(
            func=f,
            variants=variants,
            constraints=constraints,
            options=options,
            register=register,
        )

    return decorator(func) if func else decorator
