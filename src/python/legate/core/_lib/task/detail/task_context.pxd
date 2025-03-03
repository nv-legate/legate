# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from .returned_python_exception cimport _ReturnedPythonException


cdef extern from "legate/task/detail/task_context.h" \
        namespace "legate::detail" nogil:
    cdef cppclass _TaskContextImpl "legate::detail::TaskContext":
        # This declaration is a lie. This actually takes a
        # ReturnedExceptionType, but Cython does not understand converting
        # constructors. But since we only ever construct Python exceptions
        # here, we can get away with telling a lie.
        void set_exception(_ReturnedPythonException)
