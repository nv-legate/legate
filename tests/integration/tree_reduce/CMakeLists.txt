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

cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)

project(tree_reduce VERSION 0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(BUILD_SHARED_LIBS ON)

cmake_path(SET LEGATE_CORE_DIR NORMALIZE "${CMAKE_CURRENT_SOURCE_DIR}/../../../")

include(${LEGATE_CORE_DIR}/cmake/legate_helper_functions.cmake)

legate_ensure_legate()

legate_add_cpp_subdirectory(src TARGET tree_reduce EXPORT tree_reduce-export)

legate_add_cffi(${CMAKE_CURRENT_SOURCE_DIR}/src/tree_reduce_cffi.h TARGET tree_reduce)
legate_default_python_install(tree_reduce EXPORT tree_reduce-export)
