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
from libcpp cimport bool

from .tuple cimport _tuple
from .typedefs cimport _Domain


cpdef bool is_iterable(object obj)

cdef _tuple[uint64_t] uint64_tuple_from_iterable(object)

cdef _Domain domain_from_iterables(object, object)
