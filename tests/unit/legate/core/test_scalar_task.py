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

import pytest

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


if __name__ == "__main__":
    import sys

    # add -s to args, we do not want pytest to capture stdout here since this
    # gobbles any C++ exceptions
    sys.exit(pytest.main(sys.argv + ["-s"]))
