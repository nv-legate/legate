#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#=============================================================================

cmake_minimum_required(VERSION 3.26.4)

list(APPEND CMAKE_MESSAGE_CONTEXT "ex1_gen")

project(legate_ex1_gen VERSION 1.0 LANGUAGES C)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

message(STATUS "Using build type: ${CMAKE_BUILD_TYPE}")

find_package(HDF5 REQUIRED)

add_executable(gen_h5_data)

target_compile_options(gen_h5_data PRIVATE -Wall -Wextra -Werror)
target_link_libraries(gen_h5_data PRIVATE HDF5::HDF5)
target_sources(gen_h5_data PRIVATE gen_h5_data.c)
