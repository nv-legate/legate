#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

list(APPEND CMAKE_MESSAGE_CONTEXT "_lib")

add_subdirectory(data)
add_subdirectory(experimental)
add_subdirectory(mapping)
add_subdirectory(operation)
add_subdirectory(partitioning)
add_subdirectory(runtime)
add_subdirectory(tuning)
add_subdirectory(type)
add_subdirectory(utilities)
add_subdirectory(task)

set(cython_sources legate_defines.pyx)

legate_create_cython_modules(SOURCES "${cython_sources}")
