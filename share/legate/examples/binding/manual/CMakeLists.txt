#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

cmake_minimum_required(VERSION 3.22.1)

project(manual_binding VERSION 1.0 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_SHARED_LIBS ON)

add_library(hello_world hello_world.cc)

include(GNUInstallDirs)

set_target_properties(hello_world
                      PROPERTIES POSITION_INDEPENDENT_CODE ON LIBRARY_OUTPUT_DIRECTORY
                                                              "${CMAKE_INSTALL_LIBDIR}"
                                 ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_INSTALL_LIBDIR}")

find_package(legate QUIET)

if(NOT legate_FOUND)
  if(DEFINED ENV{LEGATE_DIR} AND DEFINED ENV{LEGATE_ARCH})
    set(legate_DIR "$ENV{LEGATE_DIR}/$ENV{LEGATE_ARCH}/cmake_build")
  endif()
  find_package(legate REQUIRED QUIET)
endif()

target_link_libraries(hello_world PUBLIC legate::legate)
