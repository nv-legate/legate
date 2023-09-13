# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from libcpp.utility cimport move

import cython


cdef extern from "timing/timing.h" namespace "legate::timing" nogil:
    cdef cppclass Time:
        int value()

    cdef Time measure_microseconds()

    cdef Time measure_nanoseconds()


cdef class PyTime:
    cdef Time _time
