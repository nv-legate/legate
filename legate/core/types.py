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

import builtins

from ._lib.type.type_info import (
    FixedArrayType,
    ReductionOp,
    StructType,
    Type,
    array_type,
    binary_type,
    bool_,
    complex64,
    complex128,
    float16,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    null_type,
    point_type,
    rect_type,
    string_type,
    struct_type,
    uint8,
    uint16,
    uint32,
    uint64,
)

__all__ = (
    "FixedArrayType",
    "ReductionOp",
    "StructType",
    "Type",
    "array_type",
    "binary_type",
    "bool_",
    "complex64",
    "complex128",
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "null_type",
    "point_type",
    "rect_type",
    "string_type",
    "struct_type",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
)


# Why is this not a member function of Dtype? Because we need to be able to
# refer to the Python int, float, bool etc. This is impossible to do from
# Cython as these names refer the the C types of the same name
def _Dtype_from_python_type(ty: type) -> Type:
    assert isinstance(ty, type)
    match ty:
        case builtins.bool:
            return bool_
        case builtins.int:
            return int64
        # The following _are_ all reachable (as evidenced by the test-suite!!)
        # but mypy doesn't see to think so. Probably because these builtins
        # also overload as functions, so by the time we reach the second
        # function mypy believes we've exhausted every possibility...
        case builtins.float:
            return float64  # type: ignore[unreachable]
        case builtins.complex:
            return complex128  # type: ignore[unreachable]
        case builtins.str:
            return string_type  # type: ignore[unreachable]
        case _:
            raise NotImplementedError(f"unsupported type: {ty}")
