# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.string cimport string as std_string


cdef extern from "legate/task/detail/returned_python_exception.h" \
      namespace "legate::detail" nogil:
    cdef cppclass _ReturnedPythonException \
          "legate::detail::ReturnedPythonException":
        _ReturnedPythonException()
        _ReturnedPythonException(const void *, size_t, std_string)
