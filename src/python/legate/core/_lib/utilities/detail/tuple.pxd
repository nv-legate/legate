# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint64_t

from ..tuple cimport _tuple
from ..typedefs cimport _DomainPoint


cdef extern from "legate/utilities/detail/tuple.h" namespace "legate::detail" nogil:  # noqa E501
    cdef _DomainPoint to_domain_point(const _tuple[uint64_t]&)
