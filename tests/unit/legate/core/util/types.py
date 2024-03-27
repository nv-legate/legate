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
