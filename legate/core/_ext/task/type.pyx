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

from collections.abc import Callable
from typing import TypeAlias

from ..._lib.data.physical_array cimport PhysicalArray
from ..._lib.data.physical_store cimport PhysicalStore
from ..._lib.task.task_context cimport TaskContext

UserFunction: TypeAlias = Callable[..., None]
VariantFunction: TypeAlias = Callable[[TaskContext], None]

cdef class InputStore(PhysicalStore):
    r"""Convenience class for specifying input stores for Legate task variants.

    This class can be used as a type annotation in order to mark parameters as
    inputs that should be taken from ``TaskContext.inputs`` when the task code
    is invoked:

    .. code-block:: python

        def task_function(in: InputStore, out: OutputStore) -> None

    """


cdef class OutputStore(PhysicalStore):
    r"""Convenience class for specifying output stores for Legate task
    variants.

    This class can be used as a type annotation in order to mark parameters as
    outputs that should be taken from ``TaskContext.outputs`` when the task
    code is invoked:

    .. code-block:: python

        def task_function(in: InputStore, out: OutputStore) -> None

    """


cdef class ReductionStore(PhysicalStore):
    r"""Convenience class for specifying reduction stores for Legate task
    variants.

    This class can be used as a type annotation in order to mark parameters as
    reductions that should be taken from ``TaskContext.reductions`` when the
    task code is invoked:

    .. code-block:: python

        def task_function(in: InputStore, red: ReductionStore) -> None

    """


cdef class InputArray(PhysicalArray):
    r"""Convenience class for specifying input arrays for Legate task variants.

    This class can be used as a type annotation in order to mark parameters as
    inputs that should be taken from ``TaskContext.inputs`` when the task code
    is invoked:

    .. code-block:: python

        def task_function(in: InputArray, out: OutputArray) -> None

    """


cdef class OutputArray(PhysicalArray):
    r"""Convenience class for specifying output arrays for Legate task
    variants.

    This class can be used as a type annotation in order to mark parameters as
    outputs that should be taken from ``TaskContext.outputs`` when the task
    code is invoked:

    .. code-block:: python

        def task_function(in: InputArray, out: OutputArray) -> None

    """
