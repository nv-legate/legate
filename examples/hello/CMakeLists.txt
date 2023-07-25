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

project(hello VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(BUILD_SHARED_LIBS ON)

find_package(legate_core REQUIRED)

legate_cpp_library_template(src TARGET hello)
legate_add_cpp_subdirectory(src TARGET hello EXPORT hello-export)

legate_add_cffi(${CMAKE_CURRENT_SOURCE_DIR}/src/hello_cffi.h TARGET hello)
legate_python_library_template(hello)
legate_default_python_install(hello EXPORT hello-export)
