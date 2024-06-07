# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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


cdef class OrderedSet:
    cdef dict[object, object] _dict

    cpdef void add(self, object obj)
    cpdef void update(self, object iterable)
    cpdef void discard(self, object obj)
    cpdef OrderedSet remove_all(self, OrderedSet other)
