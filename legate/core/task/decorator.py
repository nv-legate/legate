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

from typing import TYPE_CHECKING, Callable, overload

if TYPE_CHECKING:
    from .type import UserFunction, VariantList

from .task import PyTask
from .util import DEFAULT_VARIANT_LIST, dynamic_docstring


@overload
def task(func: UserFunction) -> PyTask:
    ...


@overload
def task(
    *,
    variants: VariantList = DEFAULT_VARIANT_LIST,
    register: bool = True,
) -> Callable[[UserFunction], PyTask]:
    ...


@dynamic_docstring(DEFAULT_VARIANT_LIST=DEFAULT_VARIANT_LIST)
def task(
    func: UserFunction | None = None,
    *,
    variants: VariantList = DEFAULT_VARIANT_LIST,
    register: bool = True,
) -> Callable[[UserFunction], PyTask] | PyTask:
    r"""Convert a Python function to a Legate task.

    Parameters
    ----------
    func : UserFunction
        The base user function to invoke in the task.
    variants : VariantList, optional
        The list of variants for which ``func`` is applicable. Defaults
        to ``{DEFAULT_VARIANT_LIST}``.
    register : bool, True
        Whether to immediately complete registration of the task.

    Returns
    -------
    task : PyTask
        The task object.

    See Also
    --------
    legate.core.task.task.PyTask.__init__
    """

    def decorator(f: UserFunction) -> PyTask:
        return PyTask(func=f, variants=variants, register=register)

    return decorator(func) if func else decorator
