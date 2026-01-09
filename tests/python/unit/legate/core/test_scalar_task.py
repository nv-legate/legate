# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
from typing import Any

import numpy as np
from numpy.typing import NDArray  # noqa: TC002

import pytest

from legate.core import (
    Scalar,
    Type,
    VariantOptions,
    get_legate_runtime,
    types as ty,
)
from legate.core.task import task

from .util.task_util import assert_isinstance


def NOT_NATIVELY_SUPPORTED_WARNING(obj: Any) -> str:
    return re.escape(
        f"Argument type: {type(obj)} not natively supported by type "
        "inference, falling back to pickling (which may incur a slight "
        "performance penalty). Consider opening a bug report at "
        "https://github.com/nv-legate/legate.core."
    )


class TestScalarTask:
    @pytest.mark.parametrize(
        ("value", "dtype", "py_dtype"),
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
        @task(options=VariantOptions(may_throw_exception=True))
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

    def test_default_args(self) -> None:
        @task
        def default_args(x_default: int, x: int = 1) -> None:
            assert_isinstance(x, int)
            assert_isinstance(x_default, int)
            assert x == x_default

        default_args(1)
        get_legate_runtime().issue_execution_fence(block=True)
        default_args(10, 10)
        get_legate_runtime().issue_execution_fence(block=True)
        default_args(10, x=10)
        get_legate_runtime().issue_execution_fence(block=True)
        default_args(x_default=10, x=10)
        get_legate_runtime().issue_execution_fence(block=True)

    def test_default_args_complex_type(self) -> None:
        @task
        def default_args_dict(
            x_default: dict[str, float],
            # The point of this test is to ensure that defaults are preserved
            x: dict[str, float] = {"foo": 1.2},  # noqa: B006
        ) -> None:
            assert_isinstance(x, dict)
            assert_isinstance(x_default, dict)
            assert x == x_default

        d: dict[str, float] = {}
        with pytest.warns(
            UserWarning, match=NOT_NATIVELY_SUPPORTED_WARNING(d)
        ):
            default_args_dict({"foo": 1.2})
        get_legate_runtime().issue_execution_fence(block=True)

        x_same = {"bar": 5.5}
        with pytest.warns(
            UserWarning, match=NOT_NATIVELY_SUPPORTED_WARNING(x_same)
        ):
            default_args_dict(x_same, x_same)
        get_legate_runtime().issue_execution_fence(block=True)

        with pytest.warns(
            UserWarning, match=NOT_NATIVELY_SUPPORTED_WARNING(x_same)
        ):
            default_args_dict(x_same, x=x_same)
        get_legate_runtime().issue_execution_fence(block=True)

        with pytest.warns(
            UserWarning, match=NOT_NATIVELY_SUPPORTED_WARNING(x_same)
        ):
            default_args_dict(x_default=x_same, x=x_same)
        get_legate_runtime().issue_execution_fence(block=True)


class TestScalarTaskBad:
    def test_mismatched_type_hint_call_scalar(self) -> None:
        @task
        def foo(x: int) -> None:
            pytest.fail("This point should never be reached")

        scal = Scalar(1, ty.int64)
        msg = re.escape(
            "Task expected a value of type <class 'int'> for parameter x, but "
            f"got {type(scal)}"
        )
        with pytest.raises(TypeError, match=msg):
            # The callsite type must match the type-hint exactly for Scalar
            # parameters.
            foo(scal)

    def test_mismatched_type_hint_call_pytype(self) -> None:
        @task
        def foo(x: Scalar) -> None:
            pytest.fail("This point should never be reached")

        arg = 1
        msg = re.escape(
            f"Task expected a value of type {Scalar!r} for parameter x, "
            f"but got {type(arg)}"
        )
        with pytest.raises(TypeError, match=msg):
            # The callsite must match the type-hint exactly for Scalar
            # parameters
            foo(arg)

    def test_empty_tuple(self) -> None:
        tup: tuple[int, ...] = ()

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
    sys.exit(pytest.main([*sys.argv, "-s"]))
