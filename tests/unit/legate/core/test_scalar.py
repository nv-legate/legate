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

import ctypes

import numpy as np
import pytest

import legate.core.types as ty
from legate.core import Scalar


@pytest.fixture
def scalar_s_list() -> list[tuple[bool | str | None, ty.Type]]:
    return [
        (None, ty.null_type),
        (False, ty.bool_),
        ("hello", ty.string_type),
    ]


@pytest.fixture
def scalar_i_list() -> list[tuple[int | float, ty.Type]]:
    return [
        (20, ty.int8),
        (-200.0, ty.int16),
        (200, ty.int32),
        (-200, ty.int64),
        (20, ty.uint8),
        (200.0, ty.uint16),
        (200, ty.uint32),
        (200, ty.uint64),
    ]


@pytest.fixture
def scalar_f_list() -> list[tuple[float, ty.Type]]:
    return [
        (300, ty.float16),
        (-300.456, ty.float32),
        (300.456, ty.float64),
    ]


@pytest.fixture
def scalar_c_list() -> list[tuple[complex, ty.Type]]:
    return [
        (-3.4 + 2.0j, ty.complex64),
        (-300.456 + 128j, ty.complex128),
    ]


@pytest.fixture
def scalar_fnp_list() -> list[tuple[float, ty.Type, np.generic]]:
    return [
        (300, ty.float16, np.float16(300)),
        (-300.456, ty.float32, np.float32(-300.456)),
        (300.456, ty.float64, np.float64(300.456)),
    ]


@pytest.fixture
def scalar_cnp_list() -> list[tuple[complex, ty.Type, np.generic]]:
    return [
        (-3.4 + 2.3j, ty.complex64, np.complex64(-3.4 + 2.3j)),
        (-300.456 + 128.2j, ty.complex128, np.complex128(-300.456 + 128.2j)),
    ]


class TestScalar:
    def test_scalar_type_str(
        self, scalar_s_list: list[tuple[bool | str | None, ty.Type]]
    ) -> None:
        for number, data_type in scalar_s_list:
            value = Scalar(number, data_type)
            assert value.type == data_type

    def test_scalar_type_int(
        self, scalar_i_list: list[tuple[int | float, ty.Type]]
    ) -> None:
        for number, data_type in scalar_i_list:
            value = Scalar(number, data_type)
            assert value.type == data_type

    def test_scalar_type_float(
        self, scalar_f_list: list[tuple[float, ty.Type]]
    ) -> None:
        for number, data_type in scalar_f_list:
            value = Scalar(number, data_type)
            assert value.type == data_type

    def test_scalar_type_complex(
        self, scalar_c_list: list[tuple[complex, ty.Type]]
    ) -> None:
        for number, data_type in scalar_c_list:
            value = Scalar(number, data_type)
            assert value.type == data_type

    def test_scalar_value_int(
        self, scalar_i_list: list[tuple[int | float, ty.Type]]
    ) -> None:
        for number, data_type in scalar_i_list:
            value = Scalar(number, data_type)
            assert value.value() == np.int64(number)

    def test_scalar_value_float(
        self, scalar_fnp_list: list[tuple[float, ty.Type, np.generic]]
    ) -> None:
        for number, data_type, np_number in scalar_fnp_list:
            value = Scalar(number, data_type)
            assert abs(value.value() - np_number) < 0.0001

    def test_scalar_value_complex(
        self, scalar_cnp_list: list[tuple[complex, ty.Type, np.generic]]
    ) -> None:
        for number_c, data_type, np_number_c in scalar_cnp_list:
            value = Scalar(number_c, data_type)
            assert abs(value.value() - np_number_c) < 0.0001

    def test_null_int(
        self, scalar_i_list: list[tuple[int | float, ty.Type]]
    ) -> None:
        for number, data_type in scalar_i_list:
            value = Scalar(number, data_type)
            assert value.null().value() is None
            assert value.null().type == ty.null_type

    def test_null_float(
        self, scalar_f_list: list[tuple[float, ty.Type]]
    ) -> None:
        for number, data_type in scalar_f_list:
            value = Scalar(number, data_type)
            assert value.null().value() is None
            assert value.null().type == ty.null_type

    def test_raw_handle(
        self, scalar_i_list: list[tuple[int | float, ty.Type]]
    ) -> None:
        for number, data_type in scalar_i_list:
            value = Scalar(number, data_type)
            assert value.raw_handle == id(value) + 24

    def test_ptr(self) -> None:
        number = Scalar(200, ty.int64)
        value = ctypes.c_int.from_address(number.ptr)
        assert value.value == 200

        number_f = Scalar(200.56, ty.float32)
        value_f = ctypes.c_float.from_address(number_f.ptr)
        assert abs(value_f.value - 200.56) < 0.0001

        number_c = Scalar(200.34 + 100j, ty.complex64)
        real_part = ctypes.c_float.from_address(number_c.ptr)
        imaginary_part = ctypes.c_float.from_address(
            number_c.ptr + ctypes.sizeof(ctypes.c_float)
        )
        assert abs(real_part.value - 200.34) < 0.0001
        assert abs(imaginary_part.value - 100) < 0.0001


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
