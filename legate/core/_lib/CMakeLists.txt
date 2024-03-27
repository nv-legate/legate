#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#=============================================================================

list(APPEND CMAKE_MESSAGE_CONTEXT "_lib")

add_subdirectory(data)
add_subdirectory(experimental)
add_subdirectory(mapping)
add_subdirectory(operation)
add_subdirectory(partitioning)
add_subdirectory(runtime)
add_subdirectory(type)
add_subdirectory(utilities)
add_subdirectory(task)

set(
  cython_sources
  legate_c.pyx
)

legate_core_create_cython_modules(SOURCES "${cython_sources}")
