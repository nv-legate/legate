# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint32_t, uint64_t, uintptr_t
from libcpp cimport bool
from libcpp.vector cimport vector as std_vector

from .tuple cimport _tuple


cpdef bool is_iterable(object obj)

# Add new types to this fused type whenever the function below needs extending
ctypedef fused AnyT:
    uint32_t
    uint64_t

cdef _tuple[AnyT] tuple_from_iterable(object, AnyT type_deduction_dummy = *)
cdef _tuple[uint64_t] uint64_tuple_from_iterable(object)


cdef inline std_vector[AnyT] std_vector_from_iterable(
    object obj, AnyT type_deduction_dummy = 0
):
    assert is_iterable(obj), f"Object {obj} is not iterable"

    cdef std_vector[AnyT] vec

    vec.reserve(len(obj))
    for v in obj:
        vec.emplace_back(<AnyT>v)
    return vec


ctypedef void (*finalizer_t)(void *)

cdef object register_finalizer(object obj, finalizer_t finalizer, void *handle)
