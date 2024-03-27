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

import pytest

import legate.core.types as ty
from legate.core import get_legate_runtime

from .util.types import _PRIMITIVES


class TestPromote:
    @pytest.mark.parametrize("dtype", _PRIMITIVES)
    @pytest.mark.parametrize("nullable", {True, False})
    def test_basic(self, dtype: ty.Type, nullable: bool) -> None:
        runtime = get_legate_runtime()
        arr = runtime.create_array(dtype, (1, 2, 3, 4), nullable)
        promoted = arr.promote(0, 5)
        assert list(promoted.extents) == [5, 1, 2, 3, 4]
        promoted = arr.promote(2, 7)
        assert list(promoted.extents) == [1, 2, 7, 3, 4]
        promoted = arr.promote(4, 100)
        assert list(promoted.extents) == [1, 2, 3, 4, 100]

    @pytest.mark.parametrize("dtype", _PRIMITIVES)
    @pytest.mark.parametrize("nullable", {True, False})
    def test_list(self, dtype: ty.Type, nullable: bool) -> None:
        runtime = get_legate_runtime()
        ty_list = ty.struct_type([dtype])
        arr = runtime.create_array(ty_list, (2, 3), nullable)
        promoted = arr.promote(1, 10)
        assert list(promoted.extents) == [2, 10, 3]

    @pytest.mark.parametrize("nullable", {True, False})
    def test_complex_list(self, nullable: bool) -> None:
        runtime = get_legate_runtime()
        ty_list = ty.struct_type([ty.int64, ty.uint16, ty.float32])
        arr = runtime.create_array(ty_list, (2, 3, 4), nullable)
        promoted = arr.promote(1, 10)
        assert list(promoted.extents) == [2, 10, 3, 4]


class TestPromoteErrors:
    @pytest.mark.parametrize("dtype", _PRIMITIVES)
    @pytest.mark.parametrize("nullable", {True, False})
    def test_invalid(self, dtype: ty.Type, nullable: bool) -> None:
        runtime = get_legate_runtime()
        arr = runtime.create_array(dtype, (1, 2, 3, 4), nullable)
        expected_msg = "Invalid promotion"
        with pytest.raises(ValueError, match=expected_msg):
            arr.promote(-1, 3)
        with pytest.raises(ValueError, match=expected_msg):
            arr.promote(5, 3)
        expected_msg = "can't convert negative value to size_t"
        with pytest.raises(OverflowError, match=expected_msg):
            arr.promote(2, -1)

    @pytest.mark.parametrize("dtype", _PRIMITIVES)
    @pytest.mark.parametrize("nullable", {True, False})
    def test_invalid_list(self, dtype: ty.Type, nullable: bool) -> None:
        runtime = get_legate_runtime()
        ty_list = ty.struct_type([dtype])
        arr = runtime.create_array(ty_list, (2, 3), nullable)
        expected_msg = "Invalid promotion"
        with pytest.raises(ValueError, match=expected_msg):
            arr.promote(3, 4)

    @pytest.mark.parametrize("nullable", {True, False})
    def test_invalid_str(self, nullable: bool) -> None:
        runtime = get_legate_runtime()
        arr = runtime.create_array(ty.string_type, {0}, nullable)
        expected_msg = "List array does not support store transformations"
        with pytest.raises(RuntimeError, match=expected_msg):
            arr.promote(1, 10)


class TestProject:
    @pytest.mark.parametrize("dtype", _PRIMITIVES)
    @pytest.mark.parametrize("nullable", {True, False})
    def test_basic(self, dtype: ty.Type, nullable: bool) -> None:
        runtime = get_legate_runtime()
        arr = runtime.create_array(dtype, (1, 2, 3, 4), nullable)
        projected = arr.project(0, 0)
        assert list(projected.extents) == [2, 3, 4]
        projected = arr.project(1, 1)
        assert list(projected.extents) == [1, 3, 4]
        projected = arr.project(3, 2)
        assert list(projected.extents) == [1, 2, 3]

    @pytest.mark.parametrize("dtype", _PRIMITIVES)
    @pytest.mark.parametrize("nullable", {True, False})
    def test_list(self, dtype: ty.Type, nullable: bool) -> None:
        runtime = get_legate_runtime()
        ty_list = ty.struct_type([dtype])
        arr = runtime.create_array(ty_list, (2, 3), nullable)
        projected = arr.project(1, 1)
        assert list(projected.extents) == [2]

    @pytest.mark.parametrize("nullable", {True, False})
    def test_complex_list(self, nullable: bool) -> None:
        runtime = get_legate_runtime()
        ty_list = ty.struct_type([ty.int64, ty.uint16, ty.float32])
        arr = runtime.create_array(ty_list, (2, 3, 4), nullable)
        projected = arr.project(1, 1)
        assert list(projected.extents) == [2, 4]


class TestProjectErrors:
    @pytest.mark.parametrize("dtype", _PRIMITIVES)
    @pytest.mark.parametrize("nullable", {True, False})
    def test_invalid_bound(self, dtype: ty.Type, nullable: bool) -> None:
        runtime = get_legate_runtime()
        arr = runtime.create_array(dtype, (1, 2, 3, 4), nullable)
        expected_msg = "Invalid projection"
        with pytest.raises(ValueError, match=expected_msg):
            arr.project(-1, 3)
        with pytest.raises(ValueError, match=expected_msg):
            arr.project(5, 3)
        expected_msg = "out of bounds"
        with pytest.raises(ValueError, match=expected_msg):
            arr.project(3, 100)
        with pytest.raises(ValueError, match=expected_msg):
            arr.project(2, -1)

    @pytest.mark.parametrize("dtype", _PRIMITIVES)
    @pytest.mark.parametrize("nullable", {True, False})
    def test_invalid_bound_list(self, dtype: ty.Type, nullable: bool) -> None:
        runtime = get_legate_runtime()
        ty_list = ty.struct_type([dtype])
        arr = runtime.create_array(ty_list, (1, 2, 3, 4), nullable)
        expected_msg = "Invalid projection"
        with pytest.raises(ValueError, match=expected_msg):
            arr.project(-1, 4)
        with pytest.raises(ValueError, match=expected_msg):
            arr.project(5, 4)
        expected_msg = "out of bounds"
        with pytest.raises(ValueError, match=expected_msg):
            arr.project(3, 100)
        with pytest.raises(ValueError, match=expected_msg):
            arr.project(2, -1)

    @pytest.mark.parametrize("nullable", {True, False})
    def test_str(self, nullable: bool) -> None:
        runtime = get_legate_runtime()
        arr = runtime.create_array(ty.string_type, {0}, nullable)
        expected_msg = "List array does not support store transformations"
        with pytest.raises(RuntimeError, match=expected_msg):
            arr.project(1, 1)


class TestSlice:
    @pytest.mark.parametrize("dtype", _PRIMITIVES)
    @pytest.mark.parametrize("nullable", {True, False})
    def test_bound(self, dtype: ty.Type, nullable: bool) -> None:
        runtime = get_legate_runtime()
        arr = runtime.create_array(dtype, (1, 2, 3, 4), nullable)
        sliced = arr.slice(2, slice(-2, -1))
        assert list(sliced.extents) == [1, 2, 1, 4]
        sliced = arr.slice(2, slice(1, 2))
        assert list(sliced.extents) == [1, 2, 1, 4]
        sliced = arr.slice(0, slice(0, 1))
        assert list(sliced.extents) == [1, 2, 3, 4]
        sliced = arr.slice(1, slice(-1, 0))
        assert list(sliced.extents) == [1, 0, 3, 4]

    @pytest.mark.parametrize("dtype", _PRIMITIVES)
    @pytest.mark.parametrize("nullable", {True, False})
    def test_bound_list(self, dtype: ty.Type, nullable: bool) -> None:
        runtime = get_legate_runtime()
        ty_list = ty.struct_type([dtype])
        arr = runtime.create_array(ty_list, (1, 2, 3, 4), nullable)
        sliced = arr.slice(2, slice(1, 2))
        assert list(sliced.extents) == [1, 2, 1, 4]
        sliced = arr.slice(2, slice(-2, -1))
        assert list(sliced.extents) == [1, 2, 1, 4]
        sliced = arr.slice(0, slice(0, 1))
        assert list(sliced.extents) == [1, 2, 3, 4]

    @pytest.mark.parametrize("nullable", {True, False})
    def test_bound_complex_list(self, nullable: bool) -> None:
        runtime = get_legate_runtime()
        ty_list = ty.struct_type([ty.int64, ty.uint16, ty.float32])
        arr = runtime.create_array(ty_list, (4, 5, 6), nullable)
        sliced = arr.slice(2, slice(-2, -1))
        assert list(sliced.extents) == [4, 5, 1]
        sliced = arr.slice(2, slice(1, 2))
        assert list(sliced.extents) == [4, 5, 1]
        sliced = arr.slice(2, slice(1, 4))
        assert list(sliced.extents) == [4, 5, 3]
        sliced = arr.slice(2, slice(3, 4))
        assert list(sliced.extents) == [4, 5, 1]
        sliced = arr.slice(0, slice(0, 1))
        assert list(sliced.extents) == [1, 5, 6]


class TestSliceErrors:
    @pytest.mark.parametrize("dtype", _PRIMITIVES)
    @pytest.mark.parametrize("nullable", {True, False})
    def test_invalid_bound(self, dtype: ty.Type, nullable: bool) -> None:
        runtime = get_legate_runtime()
        arr = runtime.create_array(dtype, (1, 2, 3, 4), nullable)
        expected_msg = "Out-of-bounds slicing"
        with pytest.raises(ValueError, match=expected_msg):
            arr.slice(2, slice(1, 4))
        with pytest.raises(ValueError, match=expected_msg):
            arr.slice(2, slice(3, 4))
        expected_msg = "Invalid slicing"
        with pytest.raises(ValueError, match=expected_msg):
            arr.slice(4, slice(1, 3))
        with pytest.raises(ValueError, match=expected_msg):
            arr.slice(-2, slice(1, 3))

    @pytest.mark.parametrize("dtype", _PRIMITIVES)
    @pytest.mark.parametrize("nullable", {True, False})
    def test_invalid_bound_list(self, dtype: ty.Type, nullable: bool) -> None:
        runtime = get_legate_runtime()
        ty_list = ty.struct_type([dtype])
        arr = runtime.create_array(ty_list, (1, 2, 3, 4), nullable)
        expected_msg = "Out-of-bounds slicing"
        with pytest.raises(ValueError, match=expected_msg):
            arr.slice(2, slice(1, 4))
        with pytest.raises(ValueError, match=expected_msg):
            arr.slice(2, slice(3, 4))
        expected_msg = "Invalid slicing"
        with pytest.raises(ValueError, match=expected_msg):
            arr.slice(4, slice(1, 3))
        with pytest.raises(ValueError, match=expected_msg):
            arr.slice(-2, slice(1, 3))

    @pytest.mark.parametrize("nullable", {True, False})
    def test_str(self, nullable: bool) -> None:
        runtime = get_legate_runtime()
        arr = runtime.create_array(ty.string_type, {0}, nullable)
        expected_msg = "List array does not support store transformations"
        with pytest.raises(RuntimeError, match=expected_msg):
            arr.slice(0, slice(0, 1))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
