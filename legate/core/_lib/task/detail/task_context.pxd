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
from .returned_python_exception cimport _ReturnedPythonException


cdef extern from "core/task/detail/task_context.h" \
        namespace "legate::detail" nogil:
    cdef cppclass _TaskContextImpl "legate::detail::TaskContext":
        # This declaration is a lie. This actually takes a
        # ReturnedExceptionType, but Cython does not understand converting
        # constructors. But since we only ever construct Python exceptions
        # here, we can get away with telling a lie.
        void set_exception(_ReturnedPythonException)
