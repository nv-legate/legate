#!/usr/bin/env python3

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

import struct
from enum import IntEnum
from typing import Any

import legate.core.types as types
from legate.core import LogicalArray, LogicalStore, get_legate_runtime

from .library import user_context as library, user_lib

legate_runtime = get_legate_runtime()


class HelloOpCode(IntEnum):
    HELLO_WORLD = user_lib.cffi.HELLO_WORLD
    SUM = user_lib.cffi.SUM
    SQUARE = user_lib.cffi.SQUARE
    IOTA = user_lib.cffi.IOTA


def print_hello(message: str) -> None:
    """Create a Legate task launch to print a message

    Args:
        message (str): The message to print
    """
    task = legate_runtime.create_auto_task(library, HelloOpCode.HELLO_WORLD)
    task.add_scalar_arg(message, types.string_type)
    task.execute()


def print_hellos(message: str, n: int) -> None:
    """Create a Legate task launch to print a message n times,
       using n replicas of the task

    Args:
        message (str): The message to print
        n (int): The number of times to print
    """
    task = legate_runtime.create_manual_task(
        library, HelloOpCode.HELLO_WORLD, [n]
    )
    task.add_scalar_arg(message, types.string_type)
    task.execute()


def _get_legate_store(input: Any) -> LogicalStore:
    """Extracts a Legate store from any object
       implementing the legete data interface

    Args:
        input (Any): The input object

    Returns:
        LogicalStore: The extracted Legate store
    """
    if isinstance(input, LogicalStore):
        return input
    if isinstance(input, LogicalArray):
        assert not (input.nullable or input.nested)
        return input.data
    data = input.__legate_data_interface__["data"]
    field = next(iter(data))
    array = data[field]
    assert not (array.nullable or array.nested)
    store = array.data
    return store


def to_scalar(input: LogicalStore) -> float:
    """Extracts a Python scalar value from a Legate store
       encapsulating a single scalar

    Args:
        input (LogicalStore): The Legate store encapsulating a scalar

    Returns:
        float: A Python scalar
    """
    # This operation blocks until the data in the LogicalStore
    # is available and correct

    # TODO: Accessors
    # print(type(input._cpp_store.get_physical_store().read_accessor()))
    # auto acc      = p_scalar.read_accessor<float, 1>();
    # float output  = static_cast<float>(acc[{0}]);
    return 0


def zero() -> LogicalStore:
    """Creates a Legate store representing a single zero scalar

    Returns:
        LogicalStore: A Legate store representing a scalar zero
    """
    data = bytearray(4)
    buf = struct.pack(f"{len(data)}s", data)
    scalar = get_legate_runtime().create_scalar(types.float32, buf)
    return legate_runtime.create_store(
        types.float32,
        shape=(1,),
        scalar=scalar,
        optimize_scalar=True,
    )


def iota(size: int) -> LogicalStore:
    """Enqueues a task that will generate a 1-D array
       1,2,...size.

    Args:
        size (int): The number of elements to generate

    Returns:
        LogicalStore: The Legate store that will hold the iota values
    """
    output = legate_runtime.create_array(
        types.float32,
        shape=(size,),
        optimize_scalar=True,
    )
    task = legate_runtime.create_auto_task(
        library,
        HelloOpCode.IOTA,
    )
    task.add_output(output)
    task.execute()
    return output


def sum(input: Any) -> LogicalStore:
    """Sums a 1-D array into a single scalar

    Args:
        input (Any): A Legate store or any object implementing
                     the Legate data interface.

    Returns:
        LogicalStore: A Legate store encapsulating the array sum
    """
    input_store = _get_legate_store(input)

    task = legate_runtime.create_manual_task(library, HelloOpCode.SUM, [1])

    # zero-initialize the output for the summation
    output = zero()

    task.add_input(input_store)
    task.add_reduction(output, types.ReductionOp.ADD)
    task.execute()
    return output


def square(input: Any) -> LogicalStore:
    """Computes the elementwise square of a 1-D array

    Args:
        input (Any): A Legate store or any object implementing
                     the Legate data interface.

    Returns:
        LogicalStore: A Legate store encapsulating a 1-D array
               holding the elementwise square values
    """
    input_store = _get_legate_store(input)

    output = legate_runtime.create_array(
        types.float32, shape=input_store.shape, optimize_scalar=True
    )
    task = legate_runtime.create_auto_task(library, HelloOpCode.SQUARE)

    task.add_input(input_store)
    task.add_output(output)
    task.add_alignment(input_store, output)
    task.execute()

    return output
