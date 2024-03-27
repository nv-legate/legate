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

from libc.stdint cimport uint64_t

from ..tuple cimport _tuple
from ..typedefs cimport _DomainPoint


cdef extern from "core/utilities/detail/tuple.h" namespace "legate::detail" nogil:  # noqa E501
    cdef _DomainPoint to_domain_point(const _tuple[uint64_t]&)
