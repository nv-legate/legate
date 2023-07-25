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

# We abuse find package for testing purposes here to
# 'find' the current build tree to test package builds


cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)

project(collective VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(BUILD_SHARED_LIBS ON)

find_package(legate_core REQUIRED)

legate_add_cpp_subdirectory(src TARGET collective  EXPORT collective-export)

legate_add_cffi(${CMAKE_CURRENT_SOURCE_DIR}/src/collective_cffi.h TARGET collective)
legate_default_python_install(collective EXPORT collective-export)
