#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

cmake_minimum_required(VERSION 3.22.1)

project(helloworld VERSION 1.0 LANGUAGES CXX)

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

add_executable(manual_tasks manual_tasks.cc)

legate_configure_target(manual_tasks)
