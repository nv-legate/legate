# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import ctypes
from typing import TYPE_CHECKING, Any

import numpy as np

import pytest

from legate.core import Type, TypeCode, types as ty

from .util.type_util import _PRIMITIVES

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


class TestType:
    @pytest.mark.parametrize(
        ("obj", "expected_ty"),
        [
            (0, ty.uint8),
            (1, ty.uint8),
            (-1, ty.int8),
            (256, ty.uint16),
            (-129, ty.int16),
            (65536, ty.uint32),
            (-32769, ty.int32),
            (4_294_967_296, ty.uint64),
            (-2_147_483_649, ty.int64),
            (True, ty.bool_),
            (False, ty.bool_),
            (None, ty.null_type),
            ("foo", ty.string_type),
            # The following always take the largest possible floating point
            # type in order to guarantee the value round-trips:
            # float(ty.floating_point(fp)) == fp must hold true.
            (1.23, ty.float64),
            (complex(1, 2), ty.complex128),
            (b"foo", ty.binary_type(3)),
            ([1, 2, 3], ty.array_type(ty.uint8, 3)),
            ((1, 2, 3), ty.array_type(ty.uint8, 3)),
            (np.array([1.0, 2.0, 3.0]), ty.array_type(ty.float64, 3)),
            (
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
                ty.array_type(ty.float32, 3),
            ),
            (np.array([None, None, None]), ty.array_type(ty.null_type, 3)),
            (np.array(1, dtype=np.int8), ty.array_type(ty.int8, 1)),
            (tuple(range(100)), ty.array_type(ty.uint8, 100)),
            (ty.int32, ty.int32),
        ],
    )
    def test_from_python_type(self, obj: Any, expected_ty: Type) -> None:
        cur_ty = Type.from_python_object(obj)
        assert cur_ty == expected_ty

    @pytest.mark.parametrize(
        ("np_dtype", "expected_ty"),
        [
            (np.dtype(np.bool_), ty.bool_),
            (np.dtype(np.int8), ty.int8),
            (np.dtype(np.int16), ty.int16),
            (np.dtype(np.int32), ty.int32),
            (np.dtype(np.int64), ty.int64),
            (np.dtype(np.uint8), ty.uint8),
            (np.dtype(np.uint16), ty.uint16),
            (np.dtype(np.uint32), ty.uint32),
            (np.dtype(np.uint64), ty.uint64),
            (np.dtype(np.float16), ty.float16),
            (np.dtype(np.float32), ty.float32),
            (np.dtype(np.float64), ty.float64),
            (np.dtype(np.complex64), ty.complex64),
            (np.dtype(np.complex128), ty.complex128),
            (np.dtype(np.str_), ty.string_type),
        ],
    )
    def test_from_numpy_dtype(
        self, np_dtype: DTypeLike, expected_ty: Type
    ) -> None:
        assert Type.from_numpy_dtype(np_dtype) == expected_ty

    def test_init_null(self) -> None:
        assert Type() == ty.null_type
        # for code coverage
        assert Type() != ""  # type: ignore[comparison-overlap]

    @pytest.mark.parametrize("dtype", _PRIMITIVES)
    def test_primitive_properties(self, dtype: Type) -> None:
        assert dtype.is_primitive is True
        assert isinstance(dtype.alignment, int)
        assert dtype.size % dtype.alignment == 0
        assert isinstance(dtype.code, int)
        assert isinstance(dtype.raw_ptr, int)
        assert isinstance(hash(dtype), int)
        assert not dtype.variable_size

    @pytest.mark.parametrize(
        ("dtype", "align", "offsets", "size"),
        [
            (
                ty.struct_type([ty.int32, ty.int8, ty.int64]),
                True,
                (0, 4, 8),
                16,
            ),
            (
                ty.struct_type([ty.int32, ty.int8, ty.int64], False),
                False,
                (0, 4, 5),
                13,
            ),
            (
                ty.struct_type([ty.array_type(ty.int32, 3)], True),
                True,
                (0,),
                12,
            ),
            (
                ty.struct_type([ty.float32, ty.binary_type(5)], True),
                True,
                # The alignment of the binary type is
                # alignof(std::max_align_t). This alignment is ultimately what
                # determines the offset. Python doesn't allow you to get that,
                # so we approximate it by getting the alignment of the largest
                # ctypes type there is. Hopefully this is portable.
                (0, ctypes.alignment(ctypes.c_longdouble)),
                # Alignment of float32 will be less than binary_type, so it
                # will be padded to alignment of binary_type. So effectively,
                # the size will be as if we had 2 binary_type's.
                2 * ctypes.alignment(ctypes.c_longdouble),
            ),
            (ty.rect_type(3), True, (0, 24), 48),
        ],
        ids=str,
    )
    def test_struct_properties(
        self,
        dtype: ty.StructType,
        align: bool,
        offsets: tuple[int, ...],
        size: int,
    ) -> None:
        assert dtype.aligned is align
        assert dtype.offsets == offsets
        assert dtype.size == size
        assert dtype.code == TypeCode.STRUCT
        max_alignment = max(
            [dtype.field_type(i).alignment for i in range(dtype.num_fields)]
        )
        assert dtype.alignment == max_alignment if dtype.aligned else 1
        assert not dtype.is_primitive
        assert not dtype.variable_size

    def test_record_reduction_op(self) -> None:
        op_kind = ty.ReductionOpKind.AND
        op_id = 34567
        ty.string_type.record_reduction_op(op_kind, op_id)
        assert ty.string_type.reduction_op_id(op_kind) == op_id


class TestTypeErrors:
    @pytest.mark.parametrize(
        "dtype", [ty.FixedArrayType, ty.StructType], ids=str
    )
    def test_unconstructable_types(self, dtype: type[Type]) -> None:
        msg = f"{dtype.__name__} objects must not be constructed directly"
        with pytest.raises(ValueError, match=msg):
            _ = dtype()

    def test_custom_type_to_numpy_dtype(self) -> None:
        class ListType(ty.Type):
            @property
            def code(self) -> TypeCode:
                return ty.TypeCode.STRUCT

        msg = re.escape(f"Invalid type code: {ty.TypeCode.STRUCT}")
        with pytest.raises(ValueError, match=msg):
            ListType().to_numpy_dtype()

    def test_unsupported_numpy_dtype(self) -> None:
        dtype = np.dtype(np.timedelta64)
        msg = re.escape(f"Unhandled numpy data type: {dtype}")
        with pytest.raises(NotImplementedError, match=msg):
            ty.Type.from_numpy_dtype(dtype)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
