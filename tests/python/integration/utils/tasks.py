# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

try:
    import cupy  # type: ignore[import-not-found]
except ModuleNotFoundError:
    cupy = None

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from numpy.typing import NDArray

from legate.core import (
    InlineAllocation,
    PhysicalStore,
    ProfileRange,
    Scalar,
    VariantCode,
    VariantOptions,
    align,
    broadcast,
)
from legate.core.task import (
    ADD,
    InputArray,
    InputStore,
    OutputArray,
    OutputStore,
    ReductionStore,
    task,
)

if TYPE_CHECKING:
    from types import ModuleType


class HasArrayInterface(Protocol):
    @property
    def __array_interface__(self) -> dict[str, Any]:
        pass

    @property
    def __cuda_array_interface__(self) -> dict[str, Any]:
        pass


ArrayConvertible: TypeAlias = HasArrayInterface | ArrayLike


def check_cupy(exc: Exception) -> None:
    if cupy is None:
        msg = "Need to install cupy for GPU variant"
        raise RuntimeError(msg) from exc


def asarray(alloc: ArrayConvertible) -> NDArray[Any]:
    try:
        arr = np.asarray(alloc)
    except ValueError as exc:
        check_cupy(exc)
        arr = cupy.asarray(alloc)
    return arr


def numpy_or_cupy(alloc: InlineAllocation) -> ModuleType:
    try:
        np.asarray(alloc)
    except ValueError as exc:
        check_cupy(exc)
        return cupy
    return np


@task(variants=tuple(VariantCode))
def basic_task() -> None:
    pass


@task(variants=tuple(VariantCode))
def zeros_task(out: OutputStore) -> None:
    lib = numpy_or_cupy(out.get_inline_allocation())
    out_arr = lib.asarray(out.get_inline_allocation())
    out_arr[:] = lib.zeros(out_arr.shape)


def _copy_store_task(
    in_store: PhysicalStore, out_store: PhysicalStore
) -> None:
    in_arr_np = asarray(in_store.get_inline_allocation())
    out_arr_np = asarray(out_store.get_inline_allocation())
    out_arr_np[:] = in_arr_np[:]


@task(
    variants=tuple(VariantCode), constraints=(align("in_store", "out_store"),)
)
def copy_store_task(in_store: InputStore, out_store: OutputStore) -> None:
    _copy_store_task(in_store, out_store)


@task(variants=tuple(VariantCode))
def copy_store_task_no_constraints(
    in_store: InputStore, out_store: OutputStore
) -> None:
    _copy_store_task(in_store, out_store)


@task(variants=tuple(VariantCode))
def partition_to_store_task(partition: InputStore, out: OutputStore) -> None:
    arr = asarray(partition.get_inline_allocation())
    out_arr = asarray(out.get_inline_allocation())
    out_arr[:] = out_arr + arr


@task(variants=tuple(VariantCode))
def copy_store_with_empty_task(
    partition: InputStore, out: OutputStore
) -> None:
    arr = asarray(partition.get_inline_allocation())
    out_empty = (
        out.domain.hi[0] < out.domain.lo[0]
        or out.domain.hi[1] < out.domain.lo[1]
    )
    if out_empty:
        return

    out_arr = asarray(out.get_inline_allocation())
    if arr.shape != out_arr.shape:
        out_arr[:] = out_arr[:] + arr[: out_arr.shape[0], : out_arr.shape[1]]
    else:
        out_arr[:] = out_arr[:] + arr[:]


@task(
    variants=tuple(VariantCode), constraints=broadcast("arg1", "arg2", "out")
)
def mixed_sum_task(
    arg1: InputArray, arg2: InputStore, out: OutputArray
) -> None:
    arr1_np = asarray(arg1.data().get_inline_allocation())
    arr2_np = asarray(arg2.get_inline_allocation())
    out_arr_np = asarray(out.data().get_inline_allocation())
    out_arr_np[:] = arr1_np + arr2_np


@task(variants=tuple(VariantCode))
def fill_task(out: OutputArray, val: Scalar) -> None:
    out_arr_np = asarray(out)
    v = val.value()
    if isinstance(v, memoryview):
        v = bytes(v)
    out_arr_np.fill(v)


@task(
    variants=tuple(VariantCode),
    constraints=(broadcast("out"),),
    options=VariantOptions(may_throw_exception=True),
)
def copy_np_array_task(
    out: OutputStore, np_arr: np.ndarray[Any, np.dtype[Any]]
) -> None:
    out_arr = asarray(out.get_inline_allocation())
    try:
        out_arr[:] = np_arr[:]
    except ValueError as exc:
        check_cupy(exc)
        out_arr[:] = cupy.asarray(np_arr)[:]


@task(variants=tuple(VariantCode))
def array_sum_task(store: InputStore, out: ReductionStore[ADD]) -> None:
    out_arr = asarray(out.get_inline_allocation())
    store_arr = asarray(store.get_inline_allocation())
    out_arr[:] = store_arr.sum()


@task(variants=tuple(VariantCode))
def repeat_task(
    store: InputStore, out: OutputStore, repeats: tuple[int, ...]
) -> None:
    lib = numpy_or_cupy(store.get_inline_allocation())
    store_arr = lib.asarray(store.get_inline_allocation())
    out_arr = lib.asarray(out.get_inline_allocation())
    for i in range(store_arr.ndim):
        store_arr = lib.repeat(store_arr, repeats[i], axis=i)
    out_arr[:] = store_arr


@task(variants=tuple(VariantCode))
def profile_range_task() -> None:
    with ProfileRange("foobarbaz"):
        pass


def basic_image_task(func_store: InputStore, range_store: InputStore) -> None:
    lib = numpy_or_cupy(func_store.get_inline_allocation())

    buf = lib.asarray(func_store.get_inline_allocation())
    try:
        func_arr = lib.frombuffer(buf, dtype=np.int64)
    except TypeError:
        func_arr = lib.frombuffer(buf.tobytes(), dtype="int64")
    range_arr = lib.asarray(range_store.get_inline_allocation())
    coords = tuple(
        func_arr[off :: range_store.ndim] for off in range(range_store.ndim)
    )

    lo = range_store.domain.lo
    shifted = tuple(coord - lo[idx] for idx, coord in enumerate(coords))

    # If any of the points are not in the range_store's domain, the
    # following indexing expression will raise an IndexError
    assert range_arr[shifted].size > 0


def basic_bloat_task(
    in_store: InputStore,
    bloat_store: InputStore,
    low_offsets: tuple[int, ...],
    high_offsets: tuple[int, ...],
    shape: tuple[int, ...],
) -> None:
    for i in range(in_store.ndim):
        assert (
            max(0, in_store.domain.lo[i] - low_offsets[i])
            == bloat_store.domain.lo[i]
        )
        assert (
            min(shape[i] - 1, in_store.domain.hi[i] + high_offsets[i])
            == bloat_store.domain.hi[i]
        )


@task(variants=tuple(VariantCode))
def fill_dlpack_task(store: OutputStore, val: int) -> None:
    lib = numpy_or_cupy(store.get_inline_allocation())
    arr = lib.from_dlpack(store)
    arr.fill(val)


@task(
    variants=tuple(VariantCode),
    constraints=(align("input1", "input2", "output"),),
)
def sum_two_inputs_task(
    input1: InputStore, input2: InputStore, output: OutputStore
) -> None:
    """Task that takes two inputs and outputs their sum."""
    lib = numpy_or_cupy(input1.get_inline_allocation())

    input1_arr = lib.asarray(input1.get_inline_allocation())
    input2_arr = lib.asarray(input2.get_inline_allocation())
    output_arr = lib.asarray(output.get_inline_allocation())

    output_arr[:] = input1_arr + input2_arr
