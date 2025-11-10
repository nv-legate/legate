# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
from typing import Any

import numpy as np

import pytest

from legate.core import (
    LEGATE_MAX_DIM,
    DimOrdering,
    Scalar,
    Shape,
    get_legate_runtime,
    types as ty,
)

from .utils.data import ARRAY_TYPES, EMPTY_SHAPES, SCALAR_VALS, SHAPES


class TestStoreCreation:
    @pytest.mark.parametrize(
        ("val", "dtype"), zip(SCALAR_VALS, ARRAY_TYPES, strict=True), ids=str
    )
    def test_create_from_numpy_scalar(self, val: Any, dtype: ty.Type) -> None:
        runtime = get_legate_runtime()
        arr_np = np.array(val, dtype=dtype.to_numpy_dtype())
        scalar = Scalar(arr_np, dtype)
        store = runtime.create_store_from_scalar(scalar)
        assert store.has_scalar_storage
        assert store.ndim == 1
        arr_store = np.asarray(
            store.get_physical_store().get_inline_allocation()
        )
        if isinstance(val, bytes):
            assert (arr_np == arr_store).all()
        else:
            assert np.allclose(arr_np, arr_store)

    @pytest.mark.parametrize("dtype", ARRAY_TYPES, ids=str)
    def test_store_dtype(self, dtype: ty.Type) -> None:
        shape = (1, 2, 3)
        runtime = get_legate_runtime()
        store = runtime.create_store(dtype=dtype, shape=shape)
        arr = np.asarray(store.get_physical_store().get_inline_allocation())

        val: bool | bytes | int
        match dtype.code:
            case ty.TypeCode.BOOL:
                val = bool(0)
            case ty.TypeCode.BINARY:
                val = b""
            case _:
                val = 0

        arr.fill(val)

        exp = np.zeros(dtype=dtype.to_numpy_dtype(), shape=shape)
        if dtype.code == ty.TypeCode.BINARY:
            assert (arr == exp).all()
        else:
            np.testing.assert_allclose(arr, exp)

    @pytest.mark.parametrize(
        "shape", [(1, 2, 3), (6,), (1, 1, 1, 1), (4096, 1, 5)], ids=str
    )
    def test_store_shape(self, shape: tuple[int, ...]) -> None:
        runtime = get_legate_runtime()
        store = runtime.create_store(dtype=ty.int32, shape=shape)
        assert store.shape == shape
        arr_store = np.asarray(
            store.get_physical_store().get_inline_allocation()
        )
        arr_np = np.empty(shape=shape)
        assert arr_np.shape == arr_store.shape

    @pytest.mark.parametrize("shape", SHAPES + EMPTY_SHAPES, ids=str)
    def test_store_shape_obj(self, shape: tuple[int, ...]) -> None:
        runtime = get_legate_runtime()
        store_shape = Shape(shape)
        store = runtime.create_store(dtype=ty.int32, shape=store_shape)
        arr_np = np.empty(shape=shape)
        for i in range(len(store.shape)):
            assert store.shape[i] == arr_np.shape[i]
        # for code coverage purposes
        assert store.shape == shape
        assert store.volume == store.shape.volume
        assert repr(store.shape) == str(store.shape)
        assert store.shape != print

    @pytest.mark.parametrize("shape", SHAPES + EMPTY_SHAPES, ids=str)
    @pytest.mark.parametrize(
        ("dtype", "val"), zip(ARRAY_TYPES, SCALAR_VALS, strict=True), ids=str
    )
    def test_create_from_numpy_array(
        self, shape: tuple[int, ...], dtype: ty.Type, val: Any
    ) -> None:
        runtime = get_legate_runtime()
        arr_np: np.ndarray[Any, Any] = np.ndarray(
            shape, dtype.to_numpy_dtype()
        )
        arr_np.fill(val)
        store = runtime.create_store_from_buffer(
            dtype, arr_np.shape, arr_np, False
        )
        arr_store = np.asarray(store.get_physical_store())
        if isinstance(val, bytes):
            assert (arr_store == arr_np).all()
        else:
            np.testing.assert_allclose(arr_np, arr_store)


class TestStoreCreationErrors:
    def test_invalid_shape_value(self) -> None:
        runtime = get_legate_runtime()
        msg = "Expected an iterable but got.*"
        with pytest.raises(ValueError, match=msg):
            runtime.create_store(ty.int32, shape=1)  # type: ignore[arg-type]

    def test_invalid_shape_type(self) -> None:
        runtime = get_legate_runtime()
        msg = "an integer is required"
        with pytest.raises(TypeError, match=msg):
            runtime.create_store(
                ty.int32,
                shape=("a", "b"),  # type:ignore [arg-type]
            )

    def test_exceed_max_dim(self) -> None:
        runtime = get_legate_runtime()
        with pytest.raises(IndexError, match="maximum number of dimensions"):
            runtime.create_store(ty.int32, shape=(1,) * (LEGATE_MAX_DIM + 1))

    def test_buffer_exceed_max_dim(self) -> None:
        runtime = get_legate_runtime()
        arr = np.empty(range(1, LEGATE_MAX_DIM + 2))
        with pytest.raises(IndexError, match="maximum number of dimensions"):
            runtime.create_store_from_buffer(ty.int32, arr.shape, arr, False)

    def test_string_scalar(self) -> None:
        runtime = get_legate_runtime()
        msg = re.escape("Store must have a fixed-size type")
        scalar = Scalar("abcd", ty.string_type)
        with pytest.raises(ValueError, match=msg):
            runtime.create_store_from_scalar(scalar)

    def test_get_unbound_physical_store(self) -> None:
        runtime = get_legate_runtime()
        msg = "Unbound store cannot be inlined mapped"
        store = runtime.create_store(ty.int64)
        with pytest.raises(ValueError, match=msg):
            store.get_physical_store()

    def test_small_buffer(self) -> None:
        runtime = get_legate_runtime()
        arr = np.array([1024])
        msg = "Passed buffer is too small for a store of shape .* and type .*"
        with pytest.raises(ValueError, match=msg):
            runtime.create_store_from_buffer(ty.uint64, (1024,), arr, False)

    def test_create_from_non_buffer(self) -> None:
        msg = re.escape(
            f"a bytes-like object is required, not '{type(print).__name__}'"
        )
        with pytest.raises(TypeError, match=msg):
            get_legate_runtime().create_store_from_buffer(
                ty.int32, (1,), print, False
            )

    @pytest.mark.xfail(reason="issue-3062")
    def test_invalid_fortran_ordering(self) -> None:
        # TODO(yimoj) [issue-3062]
        # The BufferError gets caught and re-raised as ValueError, same one
        # as test_small_buffer
        msg = re.escape(
            "Buffer expected to be Fortran order but is not F-Contiguous."
        )
        runtime = get_legate_runtime()
        shape = (2, 3, 4)
        data = np.arange(24, dtype=np.int32).reshape(shape).copy(order="K")
        with pytest.raises(BufferError, match=msg):
            runtime.create_store_from_buffer(
                ty.int32,
                shape,
                data,
                read_only=True,
                ordering=DimOrdering.fortran_order(),
            )

    @pytest.mark.xfail(reason="issue-3062")
    def test_invalid_c_ordering(self) -> None:
        # TODO(yimoj) [issue-3062]
        # The BufferError gets caught and re-raised as ValueError, same one
        # as test_small_buffer
        msg = re.escape(
            "Buffer expected to be C order but is not C-Contiguous."
        )
        runtime = get_legate_runtime()
        shape = (2, 3, 4)
        data = np.arange(24, dtype=np.int32).reshape(shape).copy(order="F")
        with pytest.raises(BufferError, match=msg):
            runtime.create_store_from_buffer(
                ty.int32,
                shape,
                data,
                read_only=True,
                ordering=DimOrdering.c_order(),
            )


class TestStoreCreationDimOrdering:
    """Test DimOrdering functionality in create_store_from_buffer."""

    def test_dim_ordering_basic(self) -> None:
        """Test basic functionality."""
        c_order = DimOrdering.c_order()
        assert c_order is not None
        assert c_order.kind == DimOrdering.Kind.C

        fortran_order = DimOrdering.fortran_order()
        assert fortran_order is not None
        assert fortran_order.kind == DimOrdering.Kind.FORTRAN

        custom_order = DimOrdering.custom_order([2, 0, 1])
        assert custom_order is not None
        assert custom_order.kind == DimOrdering.Kind.CUSTOM

    @pytest.mark.parametrize("read_only", [True, False], ids=str)
    def test_c_ordering(self, read_only: bool) -> None:
        """Test C ordering with create_store_from_buffer."""
        runtime = get_legate_runtime()

        shape = (3, 4)
        data = np.arange(12, dtype=np.int32).reshape(shape)
        store_c = runtime.create_store_from_buffer(
            ty.int32,
            shape,
            data,
            read_only=read_only,
            ordering=DimOrdering.c_order(),
        )

        assert store_c.shape == shape
        assert store_c.type == ty.int32

    @pytest.mark.parametrize("read_only", [True, False], ids=str)
    def test_fortran_ordering(self, read_only: bool) -> None:
        """Test Fortran ordering with create_store_from_buffer."""
        runtime = get_legate_runtime()

        shape = (3, 4)
        data = np.arange(12, dtype=np.int32).reshape(shape).copy(order="F")
        store_f = runtime.create_store_from_buffer(
            ty.int32,
            shape,
            data,
            read_only=read_only,
            ordering=DimOrdering.fortran_order(),
        )

        assert store_f.shape == shape
        assert store_f.type == ty.int32

    @pytest.mark.parametrize("read_only", [True, False], ids=str)
    def test_default_ordering(self, read_only: bool) -> None:
        """Test that default ordering is C order."""
        runtime = get_legate_runtime()

        shape = (2, 3)
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        store_default = runtime.create_store_from_buffer(
            ty.int32,
            shape,
            data,
            read_only=read_only,
            ordering=DimOrdering.custom_order([1, 0]),
        )
        store_c = runtime.create_store_from_buffer(
            ty.int32,
            shape,
            data,
            read_only=read_only,
            ordering=DimOrdering.c_order(),
        )

        assert store_default.shape == shape
        assert store_c.shape == shape

    @pytest.mark.parametrize("read_only", [True, False], ids=str)
    def test_custom_ordering(self, read_only: bool) -> None:
        """Test custom dimension ordering."""
        runtime = get_legate_runtime()

        shape = (2, 3, 4)
        data = np.arange(24, dtype=np.int32).reshape(shape)
        custom_order = DimOrdering.custom_order([2, 1, 0])  # Reverse order
        store_custom = runtime.create_store_from_buffer(
            ty.int32, shape, data, read_only=read_only, ordering=custom_order
        )

        assert store_custom.shape == shape
        assert store_custom.type == ty.int32

    @pytest.mark.parametrize("read_only", [True, False], ids=str)
    def test_different_data_types_with_ordering(self, read_only: bool) -> None:
        """Test ordering with different data types."""
        runtime = get_legate_runtime()

        shape = (2, 3)

        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        store_float_c = runtime.create_store_from_buffer(
            ty.float64,
            shape,
            data,
            read_only=read_only,
            ordering=DimOrdering.c_order(),
        )

        data = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64, order="F"
        )
        store_float_f = runtime.create_store_from_buffer(
            ty.float64,
            shape,
            data,
            read_only=read_only,
            ordering=DimOrdering.fortran_order(),
        )

        assert store_float_c.shape == shape
        assert store_float_f.shape == shape
        assert store_float_c.type == ty.float64
        assert store_float_f.type == ty.float64

    @pytest.mark.parametrize("shape", SHAPES, ids=str)
    def test_ordering_with_different_shapes(
        self, shape: tuple[int, ...]
    ) -> None:
        """Test ordering with various shapes."""
        runtime = get_legate_runtime()

        data = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
        store_c = runtime.create_store_from_buffer(
            ty.int32,
            shape,
            data,
            read_only=False,
            ordering=DimOrdering.c_order(),
        )
        assert store_c.shape == shape

        data = (
            np.arange(np.prod(shape), dtype=np.int32)
            .reshape(shape)
            .copy(order="F")
        )
        store_f = runtime.create_store_from_buffer(
            ty.int32,
            shape,
            data,
            read_only=False,
            ordering=DimOrdering.fortran_order(),
        )
        assert store_f.shape == shape

    @pytest.mark.parametrize("shape", SHAPES, ids=str)
    @pytest.mark.parametrize("read_only", [True, False], ids=str)
    def test_array_equal(
        self, shape: tuple[int, ...], read_only: bool
    ) -> None:
        runtime = get_legate_runtime()

        arr_np = np.random.rand(int(np.prod(shape))).reshape(shape)
        arr_physical_store = runtime.create_store_from_buffer(
            ty.float64,
            shape,
            arr_np,
            read_only=read_only,
            ordering=DimOrdering.c_order(),
        ).get_physical_store()
        assert np.array_equal(np.asarray(arr_physical_store), arr_np)

        arr_np = (
            np.random.rand(int(np.prod(shape))).reshape(shape).copy(order="F")
        )
        arr_physical_store = runtime.create_store_from_buffer(
            ty.float64,
            shape,
            arr_np,
            read_only=read_only,
            ordering=DimOrdering.fortran_order(),
        ).get_physical_store()
        assert np.array_equal(np.asarray(arr_physical_store), arr_np)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
