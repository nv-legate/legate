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

from enum import IntEnum, unique

import legate.core._lib.types as ext  # type: ignore[import]


@unique
class ReductionOp(IntEnum):
    ADD = ext.ADD
    SUB = ext.SUB
    MUL = ext.MUL
    DIV = ext.DIV
    MAX = ext.MAX
    MIN = ext.MIN
    OR = ext.OR
    AND = ext.AND
    XOR = ext.XOR


Dtype = ext.Dtype
FixedArrayDtype = ext.FixedArrayDtype
StructDtype = ext.StructDtype


null = Dtype.primitive_type(ext.NIL)
bool_ = Dtype.primitive_type(ext.BOOL)
int8 = Dtype.primitive_type(ext.INT8)
int16 = Dtype.primitive_type(ext.INT16)
int32 = Dtype.primitive_type(ext.INT32)
int64 = Dtype.primitive_type(ext.INT64)
uint8 = Dtype.primitive_type(ext.UINT8)
uint16 = Dtype.primitive_type(ext.UINT16)
uint32 = Dtype.primitive_type(ext.UINT32)
uint64 = Dtype.primitive_type(ext.UINT64)
float16 = Dtype.primitive_type(ext.FLOAT16)
float32 = Dtype.primitive_type(ext.FLOAT32)
float64 = Dtype.primitive_type(ext.FLOAT64)
complex64 = Dtype.primitive_type(ext.COMPLEX64)
complex128 = Dtype.primitive_type(ext.COMPLEX128)
string = Dtype.string_type()


def binary_type(size: int) -> Dtype:
    return Dtype.binary_type(size)


def array_type(element_type: Dtype, N: int) -> FixedArrayDtype:
    return Dtype.fixed_array_type(element_type, N)


def struct_type(field_types: list[Dtype], align: bool = False) -> StructDtype:
    return Dtype.struct_type(field_types, align)
