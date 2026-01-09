# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Generic, TypeAlias, TypeVar

from ..._lib.data.physical_array cimport PhysicalArray
from ..._lib.data.physical_store cimport PhysicalStore

from ..._lib.data.physical_array import PhysicalArray as PyPhysicalArray
from ..._lib.data.physical_store import PhysicalStore as PyPhysicalStore

from ..._lib.task.task_context cimport TaskContext

from ..._lib.type.types import ReductionOpKind

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


_T = TypeVar("_T", bound=ReductionOpKind)

cdef void add_redop_types():
    current_module = sys.modules[__name__]
    for redop in ReductionOpKind:
        setattr(current_module, redop.name, redop)

add_redop_types()


class ReductionStore(PyPhysicalStore, Generic[_T]):
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


class ReductionArray(PyPhysicalArray, Generic[_T]):
    r"""Convenience class for specifying reduction arrays for Legate task
    variants.

    This class can be used as a type annotation in order to mark parameters as
    reductions that should be taken from ``TaskContext.reductions`` when the
    task code is invoked:

    .. code-block:: python

        def task_function(in: InputArray, red: ReductionArray) -> None

    """
