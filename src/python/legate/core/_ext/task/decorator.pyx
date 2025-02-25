# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
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
    throws_exception: bool = False,
    has_side_effect: bool = False,
    register: bool = True,
) -> Callable[[UserFunction], PyTask] | PyTask:
    r"""Convert a Python function to a Legate task.

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
    throws_exception : bool, False
        True if any variants of ``func`` throws an exception, False
        otherwise.
    has_side_effect : bool, False
        Whether the task has any global side-effects. See ``AutoTask.set_side_
        effect()`` for further information.
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

        from legate.core import broadcast, align, VariantCode
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
            throws_exception=True,
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
            throws_exception=throws_exception,
            has_side_effect=has_side_effect,
            register=register,
        )

    return decorator(func) if func else decorator
