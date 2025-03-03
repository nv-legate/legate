# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from ..._lib.utilities.typedefs cimport VariantCode

cpdef void validate_variant(VariantCode kind)

cdef str _get_callable_name(object obj)
