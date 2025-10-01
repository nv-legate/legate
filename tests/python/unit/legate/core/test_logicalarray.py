# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from legate.core import (
    LEGATE_MAX_DIM,
    LogicalArray,
    get_legate_runtime,
    types as ty,
)

from .util.type_util import _PRIMITIVES


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


class TestTranspose:
    @pytest.mark.parametrize("dtype", _PRIMITIVES)
    @pytest.mark.parametrize("nullable", {True, False})
    def test_basic(self, dtype: ty.Type, nullable: bool) -> None:
        runtime = get_legate_runtime()
        arr = runtime.create_array(dtype, (1, 2, 3, 4), nullable)
        transposed = arr.transpose((1, 0, 3, 2))
        assert list(transposed.extents) == [2, 1, 4, 3]
        transposed = arr.transpose((3, 2, 1, 0))
        assert list(transposed.extents) == [4, 3, 2, 1]

    @pytest.mark.parametrize("dtype", _PRIMITIVES)
    @pytest.mark.parametrize("nullable", {True, False})
    def test_list(self, dtype: ty.Type, nullable: bool) -> None:
        runtime = get_legate_runtime()
        ty_list = ty.struct_type([dtype])
        arr = runtime.create_array(ty_list, (2, 3), nullable)
        transposed = arr.transpose((1, 0))
        assert list(transposed.extents) == [3, 2]

    @pytest.mark.parametrize("nullable", {True, False})
    def test_complex_list(self, nullable: bool) -> None:
        runtime = get_legate_runtime()
        ty_list = ty.struct_type([ty.int64, ty.uint16, ty.float32])
        arr = runtime.create_array(ty_list, (2, 3, 4), nullable)
        transposed = arr.transpose((2, 1, 0))
        assert list(transposed.extents) == [4, 3, 2]


class TestTransposeErrors:
    @pytest.mark.parametrize("dtype", _PRIMITIVES)
    @pytest.mark.parametrize("nullable", {True, False})
    def test_invalid_bound(self, dtype: ty.Type, nullable: bool) -> None:
        runtime = get_legate_runtime()
        arr = runtime.create_array(dtype, (1, 2, 3, 4), nullable)
        expected_msg = "Dimension Mismatch"
        with pytest.raises(ValueError, match=expected_msg):
            # Dimension Mismatch: expected 4 axes, but got 2
            arr.transpose((1, 3))
        with pytest.raises(ValueError, match=expected_msg):
            # Dimension Mismatch: expected 4 axes, but got 5
            arr.transpose((0, 1, 2, 3, 4))
        with pytest.raises(ValueError, match=expected_msg):
            # Dimension Mismatch: expected 4 axes, but got 2
            arr.transpose((-2, 100))
        expected_msg = "Duplicate axes found"
        with pytest.raises(ValueError, match=expected_msg):
            # ValueError: Duplicate axes found
            arr.transpose((0, 0, 1, 1))
        expected_msg = "Invalid axis"
        with pytest.raises(ValueError, match=expected_msg):
            # ValueError: Invalid axis -1 for a 4-D store
            arr.transpose((-1, 3, 2, 0))

    @pytest.mark.parametrize("dtype", _PRIMITIVES)
    @pytest.mark.parametrize("nullable", {True, False})
    def test_invalid_list(self, dtype: ty.Type, nullable: bool) -> None:
        runtime = get_legate_runtime()
        ty_list = ty.struct_type([dtype])
        arr = runtime.create_array(ty_list, (1, 2, 3, 4), nullable)
        expected_msg = "Dimension Mismatch"
        with pytest.raises(ValueError, match=expected_msg):
            # Dimension Mismatch: expected 4 axes, but got 2
            arr.transpose((1, 3))
        with pytest.raises(ValueError, match=expected_msg):
            # Dimension Mismatch: expected 4 axes, but got 5
            arr.transpose((0, 1, 2, 3, 4))
        with pytest.raises(ValueError, match=expected_msg):
            # Dimension Mismatch: expected 4 axes, but got 2
            arr.transpose((-1, 100))
        expected_msg = "Duplicate axes found"
        with pytest.raises(ValueError, match=expected_msg):
            # ValueError: Duplicate axes found
            arr.transpose((0, 0, 1, 1))
        expected_msg = "Invalid axis"
        with pytest.raises(ValueError, match=expected_msg):
            # ValueError: Invalid axis -1 for a 4-D store
            arr.transpose((-1, 3, 2, 0))

    @pytest.mark.parametrize("nullable", {True, False})
    def test_str(self, nullable: bool) -> None:
        runtime = get_legate_runtime()
        arr = runtime.create_array(ty.string_type, {0}, nullable)
        expected_msg = "List array does not support store transformations"
        with pytest.raises(RuntimeError, match=expected_msg):
            arr.transpose((0,))

    def test_non_iterable(self) -> None:
        runtime = get_legate_runtime()
        ty_list = ty.struct_type([ty.int32])
        arr = runtime.create_array(ty_list, (1, 2, 3, 4))
        # string is considered an iterable, though it won't work later
        # use something ridiculous here to get to the exception
        axes = print
        expected_msg = f"Expected an iterable but got {type(axes)}"
        with pytest.raises(ValueError, match=expected_msg):
            arr.transpose(axes)  # type: ignore[arg-type]


class TestDelinearize:
    @pytest.mark.parametrize("dtype", _PRIMITIVES)
    @pytest.mark.parametrize("nullable", {True, False})
    def test_basic(self, dtype: ty.Type, nullable: bool) -> None:
        runtime = get_legate_runtime()
        arr = runtime.create_array(dtype, (1, 2, 3, 4), nullable)
        delinearized = arr.delinearize(0, (1, 1))
        assert list(delinearized.extents) == [1, 1, 2, 3, 4]

        delinearized = arr.delinearize(3, (2, 1, 2, 1))
        assert list(delinearized.extents) == [1, 2, 3, 2, 1, 2, 1]

    @pytest.mark.parametrize("dtype", _PRIMITIVES)
    @pytest.mark.parametrize("nullable", {True, False})
    def test_list(self, dtype: ty.Type, nullable: bool) -> None:
        runtime = get_legate_runtime()
        ty_list = ty.struct_type([dtype])
        arr = runtime.create_array(ty_list, (2, 3), nullable)
        delinearized = arr.delinearize(1, (3, 1))
        assert list(delinearized.extents) == [2, 3, 1]

    @pytest.mark.parametrize("nullable", {True, False})
    def test_complex_list(self, nullable: bool) -> None:
        runtime = get_legate_runtime()
        ty_list = ty.struct_type([ty.int64, ty.uint16, ty.float32])
        arr = runtime.create_array(ty_list, (1, 2, 3, 4), nullable)
        delinearized = arr.delinearize(0, (1, 1))
        assert list(delinearized.extents) == [1, 1, 2, 3, 4]
        delinearized = arr.delinearize(2, (3, 1))
        assert list(delinearized.extents) == [1, 2, 3, 1, 4]


class TestdDelinearizeErrors:
    @pytest.mark.parametrize("dtype", _PRIMITIVES)
    @pytest.mark.parametrize("nullable", {True, False})
    def test_invalid_bound(self, dtype: ty.Type, nullable: bool) -> None:
        runtime = get_legate_runtime()
        arr = runtime.create_array(dtype, (1, 2, 3, 4), nullable)
        expected_msg = "Invalid delinearization"
        with pytest.raises(ValueError, match=expected_msg):
            arr.delinearize(4, (3, 1))

        expected_msg = "cannot be delinearized into"
        with pytest.raises(ValueError, match=expected_msg):
            arr.delinearize(3, (2,))
        with pytest.raises(ValueError, match=expected_msg):
            arr.delinearize(0, (1, 2))
        with pytest.raises(ValueError, match=expected_msg):
            arr.delinearize(3, (2, 1, 0))

    @pytest.mark.parametrize("dtype", _PRIMITIVES)
    @pytest.mark.parametrize("nullable", {True, False})
    def test_invalid_bound_list(self, dtype: ty.Type, nullable: bool) -> None:
        runtime = get_legate_runtime()
        ty_list = ty.struct_type([dtype])
        arr = runtime.create_array(ty_list, (1, 2, 3, 4), nullable)
        expected_msg = "Invalid delinearization on dimension"
        with pytest.raises(ValueError, match=expected_msg):
            arr.delinearize(4, (1, 2))
        expected_msg = "cannot be delinearized"
        with pytest.raises(ValueError, match=expected_msg):
            arr.delinearize(0, (1, 2))
        with pytest.raises(ValueError, match=expected_msg):
            arr.delinearize(2, (2,))

    @pytest.mark.parametrize("nullable", {True, False})
    def test_str(self, nullable: bool) -> None:
        runtime = get_legate_runtime()
        arr = runtime.create_array(ty.string_type, {6}, nullable)
        expected_msg = "List array does not support store transformations"
        with pytest.raises(RuntimeError, match=expected_msg):
            arr.delinearize(0, (6,))

    def test_non_iterable(self) -> None:
        runtime = get_legate_runtime()
        ty_list = ty.struct_type([ty.int32])
        arr = runtime.create_array(ty_list, (1, 2, 3, 4))
        # string is considered an iterable, though it won't work later
        # use something ridiculous here to get to the exception
        axes = print
        expected_msg = f"Expected an iterable but got {type(axes)}"
        with pytest.raises(ValueError, match=expected_msg):
            arr.delinearize(0, axes)  # type: ignore[arg-type]


class TestFromRawHandle:
    @pytest.mark.parametrize("dtype", _PRIMITIVES)
    @pytest.mark.parametrize("nullable", {True, False})
    def test_basic(self, dtype: ty.Type, nullable: bool) -> None:
        runtime = get_legate_runtime()
        arr = runtime.create_array(dtype, (1, 2, 3, 4), nullable)
        arr2 = LogicalArray.from_raw_handle(arr.raw_handle)
        assert list(arr2.extents) == [1, 2, 3, 4]
        assert arr2.type == dtype

    @pytest.mark.parametrize("dtype", _PRIMITIVES)
    @pytest.mark.parametrize("nullable", {True, False})
    def test_list(self, dtype: ty.Type, nullable: bool) -> None:
        runtime = get_legate_runtime()
        ty_list = ty.struct_type([dtype])
        arr = runtime.create_array(ty_list, (1, 2, 3, 4), nullable)
        arr2 = LogicalArray.from_raw_handle(arr.raw_handle)

        assert arr2.num_children == 1
        assert arr2.child(0).type == dtype
        assert list(arr2.child(0).extents) == [1, 2, 3, 4]

    @pytest.mark.parametrize("nullable", {True, False})
    def test_complex_list(self, nullable: bool) -> None:
        runtime = get_legate_runtime()
        ty_list = ty.struct_type([ty.int64, ty.uint16, ty.float32])
        arr = runtime.create_array(ty_list, (5, 6, 7), nullable)
        raw_arr = LogicalArray.from_raw_handle(arr.raw_handle)

        assert list(raw_arr.extents) == [5, 6, 7]
        assert list(raw_arr.child(2).extents) == [5, 6, 7]
        assert raw_arr.child(0).type == ty.int64
        assert raw_arr.child(1).type == ty.uint16
        assert raw_arr.child(2).type == ty.float32


class TestFromLogicalStore:
    @pytest.mark.parametrize("dtype", _PRIMITIVES)
    @pytest.mark.parametrize("optimize_scalar", {True, False})
    @pytest.mark.parametrize("ndim", {2, 3})
    def test_basic(
        self, dtype: ty.Type, optimize_scalar: bool, ndim: int
    ) -> None:
        runtime = get_legate_runtime()
        arr_store = runtime.create_store(
            dtype=dtype, optimize_scalar=optimize_scalar, ndim=ndim
        )
        arr = LogicalArray.from_store(arr_store)
        assert arr.ndim == ndim
        assert arr.type == dtype

    @pytest.mark.parametrize("dtype", _PRIMITIVES)
    @pytest.mark.parametrize("optimize_scalar", {True, False})
    @pytest.mark.parametrize("shape", {(2,), (1, 2, 3, 4)})
    def test_shape(
        self, dtype: ty.Type, optimize_scalar: bool, shape: tuple[int]
    ) -> None:
        runtime = get_legate_runtime()
        arr_store = runtime.create_store(
            dtype=dtype, shape=shape, optimize_scalar=optimize_scalar
        )
        arr = LogicalArray.from_store(arr_store)
        assert arr.shape == shape
        assert arr.type == dtype
        assert list(arr.extents) == list(shape)


class TestChild:
    @pytest.mark.parametrize("dtype", _PRIMITIVES)
    @pytest.mark.parametrize("nullable", {True, False})
    def test_basic(self, dtype: ty.Type, nullable: bool) -> None:
        runtime = get_legate_runtime()
        arr = runtime.create_array(dtype, (1, 2, 3, 4), nullable)
        assert arr.num_children == 0
        ty_list = ty.struct_type([dtype])
        arr_list = runtime.create_array(ty_list, (2, 3), nullable)
        assert arr_list.num_children == 1
        assert arr_list.child(0).type == dtype
        assert list(arr_list.child(0).extents) == [2, 3]

    @pytest.mark.parametrize("nullable", {True, False})
    def test_complex_list(self, nullable: bool) -> None:
        runtime = get_legate_runtime()
        ty_list = ty.struct_type([ty.int64, ty.uint16, ty.float32])
        arr = runtime.create_array(ty_list, (1, 2, 3, 4), nullable)
        assert arr.num_children == 3
        assert list(arr.child(0).extents) == [1, 2, 3, 4]
        assert list(arr.child(1).extents) == [1, 2, 3, 4]
        assert list(arr.child(2).extents) == [1, 2, 3, 4]
        assert arr.child(0).type == ty.int64
        assert arr.child(1).type == ty.uint16
        assert arr.child(2).type == ty.float32


class TestCreate:
    def test_unbound_array(self) -> None:
        dtype = ty.float16
        runtime = get_legate_runtime()
        arr = runtime.create_array(dtype)
        assert arr.unbound
        assert arr.ndim == 1
        # unbound shape comparison will abort python proc, so just check str
        # arr2 = runtime.create_array(ty.int32)
        # arr.shape == arr2.shape
        # terminate called after throwing an instance of
        # 'std::invalid_argument'
        #   what():  Illegal to access an uninitialized unbound store
        assert str(arr.shape) == "Shape(unbound 1D)"
        assert arr.type == dtype
        assert not arr.nullable
        assert not arr.nested
        assert arr.num_children == 0

    def test_string_type_unbound(self) -> None:
        dtype = ty.string_type
        runtime = get_legate_runtime()
        arr = runtime.create_array(dtype, (1,))
        assert arr.unbound
        assert arr.ndim == 1
        assert arr.shape == (1,)
        assert arr.type == dtype
        assert not arr.nullable
        assert arr.nested
        assert arr.num_children == 2

    def test_array_ndim(self) -> None:
        dtype = ty.int32
        runtime = get_legate_runtime()
        arr = runtime.create_array(dtype=ty.int32, ndim=2)
        assert arr.unbound
        assert arr.ndim == 2
        assert str(arr.shape) == "Shape(unbound 2D)"
        assert arr.type == dtype
        assert not arr.nullable
        assert not arr.nested
        assert arr.num_children == 0


class TestCreateErrors:
    def test_string_type_ndim(self) -> None:
        runtime = get_legate_runtime()
        msg = "List/string arrays can only be 1D"
        with pytest.raises(ValueError, match=msg):
            runtime.create_array(ty.string_type, (1, 1, 1))

    def test_exceed_max_ndim(self) -> None:
        runtime = get_legate_runtime()
        msg = "maximum number of dimensions"
        with pytest.raises(IndexError, match=msg):
            runtime.create_array(ty.int8, ndim=LEGATE_MAX_DIM + 1)

    def test_shape_and_ndim(self) -> None:
        runtime = get_legate_runtime()
        with pytest.raises(ValueError, match="ndim cannot be used with shape"):
            runtime.create_array(ty.int8, shape=(100, 1), ndim=2)

    def test_access_unbound_properties(self) -> None:
        runtime = get_legate_runtime()
        arr = runtime.create_array(ty.float16)
        msg = "Illegal to access an uninitialized unbound store"
        with pytest.raises(ValueError, match=msg):
            _ = arr.extents
        with pytest.raises(ValueError, match=msg):
            _ = arr.volume
        with pytest.raises(ValueError, match=msg):
            _ = arr.size

    def test_getitem_from_nullable(self) -> None:
        runtime = get_legate_runtime()
        msg = "Indexing is not implemented for nested or nullable arrays"
        arr = runtime.create_array(ty.int32, (1, 2, 3), nullable=True)
        with pytest.raises(NotImplementedError, match=msg):
            arr[0]
        arr = runtime.create_array(ty.string_type, (1,))
        with pytest.raises(NotImplementedError, match=msg):
            arr[0]

    def test_null_mask_non_nullable(self) -> None:
        runtime = get_legate_runtime()
        msg = "Invalid to retrieve the null mask of a non-nullable array"
        arr = runtime.create_array(ty.int32, (1, 2, 3), nullable=False)
        with pytest.raises(ValueError, match=msg):
            _ = arr.null_mask


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
