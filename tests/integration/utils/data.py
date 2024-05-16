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

from legate.core import LEGATE_MAX_DIM, types as ty

ARRAY_TYPES = (
    ty.bool_,
    ty.complex128,
    ty.complex64,
    ty.float16,
    ty.float32,
    ty.float64,
    ty.int16,
    ty.int32,
    ty.int64,
    ty.int8,
    ty.uint16,
    ty.uint32,
    ty.uint64,
    ty.uint8,
)

SCALAR_VALS = (
    True,
    complex(1, 5),
    complex(5, 1),
    12.5,
    3.1415,
    0.7777777,
    10,
    1024,
    4096,
    -1,
    65535,
    4294967295,
    101010,
)

SHAPES = (
    (1, 3, 1),
    (3, 1, 3),
    (2, 1024, 1),
    (1024,),
    (3, 6, 9),
    tuple(range(1, LEGATE_MAX_DIM + 1)),
)

EMPTY_SHAPES = ((1, 0, 3), (0, 0, 0), (1024, 0), (0,))
