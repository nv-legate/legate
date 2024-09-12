# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from enum import IntEnum
from typing import Any

import numpy as np

from ..utilities.typedefs import GlobalRedopID

class TypeCode(IntEnum):
    NIL: int
    BOOL: int
    INT8: int
    INT16: int
    INT32: int
    INT64: int
    UINT8: int
    UINT16: int
    UINT32: int
    UINT64: int
    FLOAT16: int
    FLOAT32: int
    FLOAT64: int
    COMPLEX64: int
    COMPLEX128: int
    BINARY: int
    FIXED_ARRAY: int
    STRUCT: int
    STRING: int
    LIST: int

class ReductionOpKind(IntEnum):
    ADD: int
    MUL: int
    MAX: int
    MIN: int
    OR: int
    AND: int
    XOR: int

class Type:
    @staticmethod
    def binary_type(size: int) -> Type: ...
    @staticmethod
    def fixed_array_type(element_type: Type, N: int) -> Type: ...
    @staticmethod
    def struct_type(field_types: list[Type], align: bool = True) -> Type: ...
    @property
    def code(self) -> TypeCode: ...
    @property
    def size(self) -> int: ...
    @property
    def uid(self) -> int: ...
    @property
    def variable_size(self) -> bool: ...
    @property
    def is_primitive(self) -> bool: ...
    def record_reduction_op(
        self,
        op_kind: ReductionOpKind | int,
        reduction_op_id: GlobalRedopID | int,
    ) -> None: ...
    def reduction_op_id(
        self, op_kind: ReductionOpKind | int
    ) -> GlobalRedopID: ...
    def __repr__(self) -> str: ...
    def to_numpy_dtype(self) -> np.dtype[Any]: ...
    @property
    def raw_ptr(self) -> int: ...

class FixedArrayType(Type):
    def num_elements(self) -> int: ...
    @property
    def element_type(self) -> Type: ...

class StructType(Type):
    def num_fields(self) -> int: ...
    def field_type(self, field_idx: int) -> Type: ...

null_type: Type
bool_: Type
int8: Type
int16: Type
int32: Type
int64: Type
uint8: Type
uint16: Type
uint32: Type
uint64: Type
float16: Type
float32: Type
float64: Type
complex64: Type
complex128: Type
string_type: Type

def binary_type(size: int) -> Type: ...
def array_type(element_type: Type, N: int) -> FixedArrayType: ...
def struct_type(
    field_types: list[Type], align: bool = False
) -> StructType: ...
def point_type(ndim: int) -> FixedArrayType: ...
def rect_type(ndim: int) -> StructType: ...
