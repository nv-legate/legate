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

import re
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from legate.core import Scalar, Type, types as ty
from legate.core.task import task

from .util.task_util import assert_isinstance


class TestScalarTask:
    @pytest.mark.parametrize(
        "value, dtype, py_dtype",
        [
            (1, ty.int64, int),
            (10, ty.int32, int),
            (1.0, ty.float64, float),
            (2.0, ty.float32, float),
            (None, ty.null_type, type(None)),
            (True, ty.bool_, bool),
            ("hello", ty.string_type, str),
        ],
    )
    def test_basic(self, value: Any, dtype: Type, py_dtype: type) -> None:
        @task
        def basic_scalar_task(x: Scalar) -> None:
            assert_isinstance(x, Scalar)
            assert x.type == dtype
            assert_isinstance(x.value(), py_dtype)
            assert x.value() == value

        scal = Scalar(value, dtype)
        basic_scalar_task(scal)

    def test_basic_from_scalar_value(self) -> None:
        val = 1234

        @task
        def foo(x: int) -> None:
            assert_isinstance(x, int)
            assert x == val

        scal = Scalar(val, ty.int32)
        scal_value = scal.value()
        foo(scal_value)

    @pytest.mark.parametrize("tup", ((1,), (1, 2, 3)))
    def test_tuple(self, tup: tuple[int, ...]) -> None:
        @task(throws_exception=True)
        def tuple_task(x: tuple[int, ...]) -> None:
            assert_isinstance(x, tuple)
            assert x == tup

        tuple_task(tup)

    @pytest.mark.parametrize("lst", ([400.0], [1.0, 2.0, 3.0]))
    def test_list(self, lst: list[float]) -> None:
        @task
        def list_task(x: list[float]) -> None:
            assert_isinstance(x, list)
            assert x == lst

        list_task(lst)

    def test_np_array(self) -> None:
        arr = np.array((1, 2, 3, 4), dtype=np.int64)

        @task
        def np_array_task(x: NDArray[np.int64]) -> None:
            assert_isinstance(x, np.ndarray)
            assert x.dtype == arr.dtype
            np.testing.assert_array_equal(x, arr)

        np_array_task(arr)


def NOT_NATIVELY_SUPPORTED_WARNING(obj: Any) -> str:
    return re.escape(
        f"Argument type: {type(obj)} not natively supported by type "
        "inference, falling back to pickling (which may incur a slight "
        "performance penalty). Consider opening a bug report at "
        "https://github.com/nv-legate/legate.core."
    )


class TestScalarTaskBad:
    def test_mismatched_type_hint_call_scalar(self) -> None:
        @task
        def foo(x: int) -> None:
            assert False, "This point should never be reached"

        scal = Scalar(1, ty.int64)
        msg = re.escape(
            "Task expected a value of type <class 'int'> for parameter x, but "
            "got <class 'legate.core._lib.data.scalar.Scalar'>"
        )
        with pytest.raises(TypeError, match=msg):
            # The callsite type must match the type-hint exactly for Scalar
            # parameters.
            foo(scal)

    def test_mismatched_type_hint_call_pytype(self) -> None:
        @task
        def foo(x: Scalar) -> None:
            assert False, "This point should never be reached"

        msg = re.escape(
            "Task expected a value of type <class "
            "'legate.core._lib.data.scalar.Scalar'> for parameter x, but got "
            "<class 'int'>"
        )
        with pytest.raises(TypeError, match=msg):
            # The callsite must match the type-hint exactly for Scalar
            # parameters
            foo(1)

    def test_empty_tuple(self) -> None:
        tup: tuple[int, ...] = tuple()

        @task
        def tuple_task(x: tuple[int, ...]) -> None:
            assert_isinstance(x, tuple)
            assert x == tup

        with pytest.warns(
            UserWarning, match=NOT_NATIVELY_SUPPORTED_WARNING(tup)
        ):
            tuple_task(tup)

    @pytest.mark.parametrize(
        "d",
        (
            {1: 1},
            {"foo": 1, 1.3: (1, 2, 4)},
            {frozenset((1, 2, 3)): {"hello": "goodbye"}},
        ),
    )
    def test_dict(self, d: dict[Any, Any]) -> None:

        @task
        def dict_task(x: dict[Any, Any]) -> None:
            assert_isinstance(x, dict)
            assert x == d

        with pytest.warns(
            UserWarning, match=NOT_NATIVELY_SUPPORTED_WARNING(d)
        ):
            dict_task(d)


if __name__ == "__main__":
    import sys

    # add -s to args, we do not want pytest to capture stdout here since this
    # gobbles any C++ exceptions
    sys.exit(pytest.main(sys.argv + ["-s"]))
