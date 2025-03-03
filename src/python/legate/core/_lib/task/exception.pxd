# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t
from libcpp.string cimport string as std_string


cdef extern from "legate/task/exception.h" namespace "legate" nogil:
    cdef cppclass _TaskException "legate::TaskException":
        _TaskException(int32_t, std_string) except+
        _TaskException(std_string) except+
        const char* what() except+
        int32_t index() except+
        std_string error_message() except+
