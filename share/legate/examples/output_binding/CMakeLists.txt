#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

cmake_minimum_required(VERSION 3.22.1)

project(output_binding VERSION 1.0 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(BUILD_SHARED_LIBS ON)

find_package(legate QUIET)

if(NOT legate_FOUND)
  if(LEGATE_DIR AND LEGATE_ARCH)
    set(legate_DIR "${LEGATE_DIR}/${LEGATE_ARCH}/cmake_build")
  elseif(DEFINED ENV{LEGATE_DIR} AND DEFINED ENV{LEGATE_ARCH})
    set(legate_DIR "$ENV{LEGATE_DIR}/$ENV{LEGATE_ARCH}/cmake_build")
  endif()
  find_package(legate REQUIRED)
endif()

add_executable(output_binding output_binding.cc)

# output_binding.cc includes the example's own header as <output_binding.h>, so the example source
# directory must be on the include path.
target_include_directories(output_binding PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

legate_configure_target(output_binding)
