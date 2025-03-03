# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from legate.core import LEGATE_MAX_DIM, types as ty

SCALAR_VALS = (
    True,  # bool
    complex(1, 5),  # complex128
    complex(5, 1),  # complex64
    12.5,  # float16
    3.1415,  # float32
    0.7777777,  # float64
    10,  # int16
    1024,  # int32
    4096,  # int64
    -1,  # int8
    65535,  # uint16
    4294967295,  # uint32
    101010,  # uint64
    123,  # uint8
    b"abcdefghijklmnopqrstuvwxyz",  # binary_type
    # None, null_type
)

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
    ty.binary_type(len(SCALAR_VALS[-1])),
    # ty.null_type
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

BROADCAST_SHAPES = (
    ((3, 1, 3), (1, 3)),
    ((5, 4), (1,)),
    ((2, 512), (512,)),
    ((15, 3, 5), (15, 1, 5)),
    ((15, 128, 5), (128, 5)),
    ((5, 1024, 5), (1024, 1)),
)

LARGE_SHAPES = (
    (5, 4096, 5),
    (8192, 1, 3),
    (2, 1024, 50),
    (1024 * 50,),
    (8,) * LEGATE_MAX_DIM,
)
