# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint64_t

from ..typedefs cimport _DomainPoint

cdef extern from "legate/utilities/tuple.h":
    cppclass _tuple "legate::tuple" [T]:
        pass

cdef extern from "legate/utilities/detail/tuple.h" nogil:  # noqa E501
    _DomainPoint _to_domain_point "legate::detail::to_domain_point" (
        const _tuple[uint64_t]&
    )
