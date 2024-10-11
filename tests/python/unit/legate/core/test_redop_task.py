# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

try:
    import cupy  # type: ignore[import-not-found]
except ModuleNotFoundError:
    cupy = None

import itertools
import re
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from numpy._typing import NDArray

from legate.core import (
    InlineAllocation,
    LogicalStore,
    get_legate_runtime,
    types as ty,
)
from legate.core.task import (
    ADD,
    AND,
    MAX,
    MIN,
    MUL,
    OR,
    XOR,
    InputStore,
    ReductionStore,
    task,
)


def check_cupy(exc: Exception) -> None:
    if cupy is None:
        raise RuntimeError("Need to install cupy for GPU variant") from exc


def asarray(alloc: InlineAllocation) -> NDArray[Any]:
    try:
        return np.asarray(alloc)
    except ValueError as exc:
        check_cupy(exc)
        return cupy.asarray(alloc)


class TestRedopTaskStore:
    def create_input_args_(
        self,
    ) -> tuple[npt.NDArray[np.int64], LogicalStore]:
        in_arr = np.arange(10, dtype=np.int64) + 1
        in_store = get_legate_runtime().create_store_from_buffer(
            ty.int64, in_arr.shape, in_arr, False
        )
        return in_arr, in_store

    def create_output_args_(
        self, init: int = 0
    ) -> tuple[npt.NDArray[np.int64], LogicalStore]:
        out_arr = np.array((init,), dtype=np.int64)
        out_store = get_legate_runtime().create_store_from_buffer(
            ty.int64, out_arr.shape, out_arr, False
        )
        return out_arr, out_store

    def test_create_bad(self) -> None:
        def without_redop_kind(
            x: ReductionStore,  # type: ignore[type-arg]
        ) -> None:
            pass

        msg = re.escape(
            "Type hint '<class 'legate.core._ext.task.type.ReductionStore'>'"
            " has an invalid number of reduction operators (0), expected 1. "
            "For example: 'x: ReductionStore[ADD]'"
        )
        with pytest.raises(TypeError, match=msg):
            task(without_redop_kind)

    def test_add(self) -> None:
        @task
        def array_sum_task(
            store: InputStore, out: ReductionStore[ADD]
        ) -> None:
            store_arr = asarray(store.get_inline_allocation())
            out_arr = asarray(out.get_inline_allocation())
            out_arr[:] = out_arr + store_arr.sum()

        in_arr, in_store = self.create_input_args_()
        out_arr, out_store = self.create_output_args_()

        array_sum_task(in_store, out_store)

        get_legate_runtime().issue_execution_fence(block=True)

        np.testing.assert_allclose(
            np.asarray(out_store.get_physical_store().get_inline_allocation()),
            np.add.reduce(in_arr),
        )

    @pytest.mark.parametrize(
        "values", tuple(itertools.product([True, False], repeat=4))
    )
    def test_and(self, values: tuple[bool, bool, bool, bool]) -> None:
        @task
        def array_and_task(
            store: InputStore, out: ReductionStore[AND]
        ) -> None:
            store_arr = asarray(store.get_inline_allocation())
            out_arr = asarray(out.get_inline_allocation())
            out_arr[:] = out_arr & np.bitwise_and.reduce(store_arr)

        runtime = get_legate_runtime()
        in_arr = np.asarray(values, dtype=np.bool_)
        in_store = runtime.create_store_from_buffer(
            ty.bool_, in_arr.shape, in_arr, False
        )
        out_arr = np.array((True,), dtype=np.bool_)
        out_store = runtime.create_store_from_buffer(
            ty.bool_, out_arr.shape, out_arr, False
        )

        array_and_task(in_store, out_store)

        runtime.issue_execution_fence(block=True)

        np.testing.assert_allclose(
            np.asarray(out_store.get_physical_store().get_inline_allocation()),
            np.bitwise_and.reduce(in_arr),
        )

    def test_max(self) -> None:
        @task
        def array_max_task(
            store: InputStore, out: ReductionStore[MAX]
        ) -> None:
            store_arr = asarray(store.get_inline_allocation())
            out_arr = asarray(out.get_inline_allocation())
            out_arr[:] = np.maximum(out_arr.max(), store_arr.max())

        in_arr, in_store = self.create_input_args_()
        out_arr, out_store = self.create_output_args_()

        array_max_task(in_store, out_store)

        get_legate_runtime().issue_execution_fence(block=True)

        np.testing.assert_allclose(
            np.asarray(out_store.get_physical_store().get_inline_allocation()),
            np.maximum.reduce(in_arr),
        )

    def test_min(self) -> None:
        @task
        def array_min_task(
            store: InputStore, out: ReductionStore[MIN]
        ) -> None:
            store_arr = asarray(store.get_inline_allocation())
            out_arr = asarray(out.get_inline_allocation())
            out_arr[:] = np.minimum(out_arr.min(), store_arr.min())

        in_arr, in_store = self.create_input_args_()
        out_arr, out_store = self.create_output_args_(init=in_arr.max() + 1)

        array_min_task(in_store, out_store)

        get_legate_runtime().issue_execution_fence(block=True)

        np.testing.assert_allclose(
            np.asarray(out_store.get_physical_store().get_inline_allocation()),
            np.minimum.reduce(in_arr),
        )

    def test_mul(self) -> None:
        @task
        def array_mul_task(
            store: InputStore, out: ReductionStore[MUL]
        ) -> None:
            store_arr = asarray(store.get_inline_allocation())
            out_arr = asarray(out.get_inline_allocation())
            out_arr[:] = out_arr * np.multiply.reduce(store_arr)

        in_arr, in_store = self.create_input_args_()
        out_arr, out_store = self.create_output_args_(init=1)

        array_mul_task(in_store, out_store)

        get_legate_runtime().issue_execution_fence(block=True)

        np.testing.assert_allclose(
            np.asarray(out_store.get_physical_store().get_inline_allocation()),
            np.multiply.reduce(in_arr),
        )

    def test_or(self) -> None:
        @task
        def array_or_task(store: InputStore, out: ReductionStore[OR]) -> None:
            store_arr = asarray(store.get_inline_allocation())
            out_arr = asarray(out.get_inline_allocation())
            out_arr[:] = out_arr | np.bitwise_or.reduce(store_arr)

        in_arr, in_store = self.create_input_args_()
        out_arr, out_store = self.create_output_args_()

        array_or_task(in_store, out_store)

        get_legate_runtime().issue_execution_fence(block=True)

        np.testing.assert_allclose(
            np.asarray(out_store.get_physical_store().get_inline_allocation()),
            np.bitwise_or.reduce(in_arr),
        )

    def test_xor(self) -> None:
        @task
        def array_xor_task(
            store: InputStore, out: ReductionStore[XOR]
        ) -> None:
            store_arr = asarray(store.get_inline_allocation())
            out_arr = asarray(out.get_inline_allocation())
            out_arr[:] = out_arr ^ np.bitwise_xor.reduce(store_arr)

        in_arr, in_store = self.create_input_args_()
        out_arr, out_store = self.create_output_args_()

        array_xor_task(in_store, out_store)

        get_legate_runtime().issue_execution_fence(block=True)

        np.testing.assert_allclose(
            np.asarray(out_store.get_physical_store().get_inline_allocation()),
            np.bitwise_xor.reduce(in_arr),
        )


if __name__ == "__main__":
    import sys

    # add -s to args, we do not want pytest to capture stdout here since this
    # gobbles any C++ exceptions
    sys.exit(pytest.main(sys.argv + ["-s"]))
