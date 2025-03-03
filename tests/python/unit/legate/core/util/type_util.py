# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import legate.core.types as ty

_PRIMITIVES = [
    ty.bool_,
    ty.int8,
    ty.int16,
    ty.int32,
    ty.int64,
    ty.uint8,
    ty.uint16,
    ty.uint32,
    ty.uint64,
    ty.float16,
    ty.float32,
    ty.float64,
    ty.complex64,
    ty.complex128,
]
