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
from libcpp.optional cimport optional as std_optional
from libcpp.string cimport string as std_string


cdef extern from "core/task/detail/task_context.h" \
        namespace "legate::detail" nogil:
    cdef cppclass _TaskContextImpl "legate::detail::TaskContext":
        void set_exception(std_string) except +
        std_optional[std_string]& get_execption() noexcept
