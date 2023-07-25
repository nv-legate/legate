# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


from typing import Any

import numpy as np

from ._legion.util import BufferBuilder

class ReductionOp:
    ADD: int
    SUB: int
    MUL: int
    DIV: int
    MAX: int
    MIN: int
    OR: int
    AND: int
    XOR: int

class Dtype:
    @staticmethod
    def fixed_array_type(element_type: Dtype, N: int) -> Dtype: ...
    @staticmethod
    def struct_type(field_types: list[Dtype], align: bool) -> Dtype: ...
    @property
    def code(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def uid(self) -> int: ...
    @property
    def variable_size(self) -> bool: ...
    @property
    def is_primitive(self) -> bool: ...
    def record_reduction_op(
        self, op_kind: int, reduction_op_id: int
    ) -> None: ...
    def reduction_op_id(self, op_kind: int) -> int: ...
    def __repr__(self) -> str: ...
    def to_numpy_dtype(self) -> np.dtype[Any]: ...
    def serialize(self, buf: BufferBuilder) -> None: ...
    @property
    def raw_ptr(self) -> int: ...

class FixedArrayDtype(Dtype):
    def num_elements(self) -> int: ...
    @property
    def element_type(self) -> Dtype: ...

class StructDtype(Dtype):
    def num_fields(self) -> int: ...
    def field_type(self, field_idx: int) -> Dtype: ...

bool_: Dtype
int8: Dtype
int16: Dtype
int32: Dtype
int64: Dtype
uint8: Dtype
uint16: Dtype
uint32: Dtype
uint64: Dtype
float16: Dtype
float32: Dtype
float64: Dtype
complex64: Dtype
complex128: Dtype
string: Dtype

def array_type(element_type: Dtype, N: int) -> FixedArrayDtype: ...
def struct_type(
    field_types: list[Dtype], align: bool = False
) -> StructDtype: ...
