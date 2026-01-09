# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

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


cdef void flatten_constraints(
    clist: Sequence[DeferredConstraint | Sequence[DeferredConstraint]],
    ret: list[DeferredConstraint]
):
    for sub in clist:
        if isinstance(sub, DeferredConstraint):
            ret.append(sub)
        else:
            flatten_constraints(clist=sub, ret=ret)


@dynamic_docstring(DEFAULT_VARIANT_LIST=DEFAULT_VARIANT_LIST)
def task(
    func: object = None,
    *,
    variants: tuple[VariantCode, ...] = DEFAULT_VARIANT_LIST,
    constraints: Sequence[
        DeferredConstraint | Sequence[DeferredConstraint]
    ] | None = None,
    options: TaskConfig | VariantOptions | None = None,
) -> Callable[[UserFunction], PyTask] | PyTask:
    r"""Convert a Python function to a Legate task.

    The user may pass either a ``TaskConfig`` or ``VariantOptions`` to select
    the task options. If the former, the user can select the task ID of the
    task themselves, while additionally setting variant options via the
    ``TaskConfig.variant_options`` property. If the latter, then a new, unique
    task ID will be automatically generated.

    Parameters
    ----------
    func : UserFunction
        The base user function to invoke in the task.
    variants : tuple[VariantCode, ...], optional
        The list of variants for which ``func`` is applicable. Defaults
        to ``{DEFAULT_VARIANT_LIST}``.
    constraints : Sequence[DeferredConstraint | Sequence[DeferredConstraint]], optional
        The list of constraints which are to be applied to the arguments of
        ``func``, if any. Defaults to no constraints.
    options : TaskConfig | VariantOptions, optional
        Either a ``TaskConfig`` or ``VariantOptions`` describing the task
        configuration.

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


    By default, the ``@task`` decorator registers the wrapped callable for
    all variants. It is possible, however, to select additional functions
    to act as specific variants. Variants must have identical calling
    signatures to the original function and inherit the same constraints and
    options.

    The original function is used for any unspecified variants.
    ::

        from legate.core.task import task

        # Acts as the CPU and OMP variant as these were not specified
        # separately
        @task
        def my_task() -> None:
            ...

        @my_task.gpu_variant
        def gpu_task_variant() -> None:
            ...


    See Also
    --------
    legate.core.task.PyTask.__init__
    """  # noqa: E501

    def decorator(f: UserFunction) -> PyTask:
        cdef list flat

        if constraints is None:
            # Preserve None-ness of the constraints. It means different things
            # to say "This task has None as constraints" vs "This task has
            # exactly zero constraints".
            flat = None
        else:
            flat = []
            flatten_constraints(clist=constraints, ret=flat)

        return PyTask(
            func=f,
            variants=variants,
            constraints=flat,
            options=options,
        )

    return decorator(func) if func else decorator
