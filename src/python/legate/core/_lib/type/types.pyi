# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum
from typing import Any, cast

import numpy as np
from numpy.typing import DTypeLike

from ..utilities.typedefs import GlobalRedopID

class TypeCode(IntEnum):
    NIL = cast(int, ...)
    BOOL = cast(int, ...)
    INT8 = cast(int, ...)
    INT16 = cast(int, ...)
    INT32 = cast(int, ...)
    INT64 = cast(int, ...)
    UINT8 = cast(int, ...)
    UINT16 = cast(int, ...)
    UINT32 = cast(int, ...)
    UINT64 = cast(int, ...)
    FLOAT16 = cast(int, ...)
    FLOAT32 = cast(int, ...)
    FLOAT64 = cast(int, ...)
    COMPLEX64 = cast(int, ...)
    COMPLEX128 = cast(int, ...)
    BINARY = cast(int, ...)
    FIXED_ARRAY = cast(int, ...)
    STRUCT = cast(int, ...)
    STRING = cast(int, ...)
    LIST = cast(int, ...)

class ReductionOpKind(IntEnum):
    ADD = cast(int, ...)
    MUL = cast(int, ...)
    MAX = cast(int, ...)
    MIN = cast(int, ...)
    OR = cast(int, ...)
    AND = cast(int, ...)
    XOR = cast(int, ...)

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
    def alignment(self) -> int: ...
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
    def to_numpy_dtype(self) -> np.dtype[Any]: ...
    @property
    def raw_ptr(self) -> int: ...
    @staticmethod
    def from_python_object(py_object: Any) -> Type: ...
    @staticmethod
    def from_numpy_dtype(dtype: DTypeLike) -> Type: ...

class FixedArrayType(Type):
    @property
    def num_elements(self) -> int: ...
    @property
    def element_type(self) -> Type: ...

class StructType(Type):
    @property
    def num_fields(self) -> int: ...
    def field_type(self, field_idx: int) -> Type: ...
    @property
    def aligned(self) -> bool: ...
    @property
    def offsets(self) -> tuple[int, ...]: ...

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
def struct_type(field_types: list[Type], align: bool = True) -> StructType: ...
def point_type(ndim: int) -> FixedArrayType: ...
def rect_type(ndim: int) -> StructType: ...
