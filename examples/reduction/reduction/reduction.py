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


from enum import IntEnum

import cunumeric as np

import legate.core.types as ty
from legate.core import LogicalStore, get_legate_runtime

from .library import user_context as library, user_lib

legate_runtime = get_legate_runtime()


class OpCode(IntEnum):
    BINCOUNT = user_lib.cffi.BINCOUNT
    CATEGORIZE = user_lib.cffi.CATEGORIZE
    HISTOGRAM = user_lib.cffi.HISTOGRAM
    MATMUL = user_lib.cffi.MATMUL
    MUL = user_lib.cffi.MUL
    SUM_OVER_AXIS = user_lib.cffi.SUM_OVER_AXIS
    UNIQUE = user_lib.cffi.UNIQUE


def _sanitize_axis(axis: int, ndim: int) -> int:
    sanitized = axis
    if sanitized < 0:
        sanitized += ndim
    if sanitized < 0 or sanitized >= ndim:
        raise ValueError(f"Invalid axis {axis} for a {ndim}-D store")
    return sanitized


def sum_over_axis(input: LogicalStore, axis: int) -> LogicalStore:
    """
    Sum values along the chosen axis

    Parameters
    ----------
    input : LogicalStore
        Input to sum
    axis : int
        Axis along which the summation should be done

    Returns
    -------
    LogicalStore
        Summation result
    """
    sanitized = _sanitize_axis(axis, input.ndim)

    # Compute the output shape by removing the axis being summed over
    res_shape = tuple(
        ext for dim, ext in enumerate(input.shape) if dim != sanitized
    )
    result = legate_runtime.create_store(input.type, res_shape)
    np.asarray(result).fill(0)

    # Broadcast the output along the contracting dimension
    promoted = result.promote(axis, input.shape[axis])

    assert promoted.shape == input.shape

    task = legate_runtime.create_auto_task(library, OpCode.SUM_OVER_AXIS)
    task.add_input(input)
    task.add_reduction(promoted, ty.ReductionOp.ADD)
    task.add_alignment(input, promoted)

    task.execute()

    return result


def multiply(rhs1: LogicalStore, rhs2: LogicalStore) -> LogicalStore:
    if rhs1.type != rhs2.type or rhs1.shape != rhs2.shape:
        raise ValueError("Stores to add must have the same type and shape")

    result = legate_runtime.create_store(rhs1.type, rhs1.shape)

    task = legate_runtime.create_auto_task(library, OpCode.MUL)
    task.add_input(rhs1)
    task.add_input(rhs2)
    task.add_output(result)
    task.add_alignment(result, rhs1)
    task.add_alignment(result, rhs2)

    task.execute()

    return result


def matmul(rhs1: LogicalStore, rhs2: LogicalStore) -> LogicalStore:
    """
    Performs matrix multiplication

    Parameters
    ----------
    rhs1, rhs2 : LogicalStore
        Matrices to multiply

    Returns
    -------
    LogicalStore
        Multiplication result
    """
    if rhs1.ndim != 2 or rhs2.ndim != 2:
        raise ValueError("Stores must be 2D")
    if rhs1.type != rhs2.type:
        raise ValueError("Stores must have the same type")
    if rhs1.shape[1] != rhs2.shape[0]:
        raise ValueError(
            "Can't do matrix mulplication between arrays of "
            f"shapes {rhs1.shape} and {rhs1.shape}"
        )

    m = rhs1.shape[0]
    k = rhs1.shape[1]
    n = rhs2.shape[1]

    # Multiplying an (m, k) matrix with a (k, n) matrix gives
    # an (m, n) matrix
    result = legate_runtime.create_store(rhs1.type, (m, n))
    np.asarray(result).fill(0)

    # Each store gets a fake dimension that it doesn't have
    rhs1 = rhs1.promote(2, n)
    rhs2 = rhs2.promote(0, m)
    lhs = result.promote(1, k)

    assert lhs.shape == rhs1.shape
    assert lhs.shape == rhs2.shape

    task = legate_runtime.create_auto_task(library, OpCode.MATMUL)
    task.add_input(rhs1)
    task.add_input(rhs2)
    task.add_reduction(lhs, ty.ReductionOp.ADD)
    task.add_alignment(lhs, rhs1)
    task.add_alignment(lhs, rhs2)

    task.execute()

    return result


def bincount(input: LogicalStore, num_bins: int) -> LogicalStore:
    """
    Counts the occurrences of each bin index

    Parameters
    ----------
    input : LogicalStore
        Input to bin-count
    num_bins : int
        Number of bins

    Returns
    -------
    LogicalStore
        Counting result
    """
    result = legate_runtime.create_store(ty.uint64, (num_bins,))
    np.asarray(result).fill(0)

    task = legate_runtime.create_auto_task(library, OpCode.BINCOUNT)
    task.add_input(input)
    # Broadcast the result store. This commands the Legate runtime to give
    # the entire store to every task instantiated by this task descriptor
    task.add_broadcast(result)
    # Declares that the tasks will do reductions to the result store and
    # that outputs from the tasks should be combined by addition
    task.add_reduction(result, ty.ReductionOp.ADD)

    task.execute()

    return result


def categorize(input: LogicalStore, bins: LogicalStore) -> LogicalStore:
    result = legate_runtime.create_store(ty.uint64, input.shape)

    task = legate_runtime.create_auto_task(library, OpCode.CATEGORIZE)
    task.add_input(input)
    task.add_input(bins)
    task.add_output(result)

    # Broadcast the store that contains bin edges. Each task will get a copy
    # of the entire bin edges
    task.add_broadcast(bins)

    task.execute()

    return result


def histogram(input: LogicalStore, bins: LogicalStore) -> LogicalStore:
    """
    Constructs a histogram for the given bins

    Parameters
    ----------
    input : LogicalStore
        Input
    bins : int
        Bin edges

    Returns
    -------
    LogicalStore
        Histogram
    """
    num_bins = bins.shape[0] - 1
    result = legate_runtime.create_store(ty.uint64, (num_bins,))
    np.asarray(result).fill(0)

    task = legate_runtime.create_auto_task(library, OpCode.HISTOGRAM)
    task.add_input(input)
    task.add_input(bins)
    task.add_reduction(result, ty.ReductionOp.ADD)

    # Broadcast both the result store and the one that contains bin edges.
    task.add_broadcast(bins)
    task.add_broadcast(result)

    task.execute()

    return result


def unique(input: LogicalStore, radix: int = 4) -> LogicalStore:
    """
    Finds unique elements in the input and returns them in a store

    Parameters
    ----------
    input : LogicalStore
        Input

    Returns
    -------
    LogicalStore
        Result that contains only the unique elements of the input
    """

    if input.ndim > 1:
        raise ValueError("`unique` accepts only 1D stores")

    if input.type in (
        ty.float16,
        ty.float32,
        ty.float64,
        ty.complex64,
        ty.complex128,
    ):
        raise ValueError(
            "`unique` doesn't support floating point or complex numbers"
        )

    # Create an unbound store to collect local results
    result = legate_runtime.create_store(input.type, shape=None, ndim=1)

    task = legate_runtime.create_auto_task(library, OpCode.UNIQUE)
    task.add_input(input)
    task.add_output(result)

    task.execute()

    # Perform global reduction using a reduction tree
    return legate_runtime.tree_reduce(
        library, OpCode.UNIQUE, result, radix=radix
    )
