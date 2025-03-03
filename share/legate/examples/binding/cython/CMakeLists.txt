#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

cmake_minimum_required(VERSION 3.22.1)

if(NOT SKBUILD)
  message(FATAL_ERROR "Required scikit-build core to run this script. See https://github.com/scikit-build/scikit-build-core"
  )
endif()

project(${SKBUILD_PROJECT_NAME} VERSION 1.0 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_SHARED_LIBS ON)

add_library(hello_world hello_world.cc)

target_include_directories(hello_world PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

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

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

add_custom_command(OUTPUT hello_world_cython.cpp
                   COMMENT "Making ${CMAKE_CURRENT_BINARY_DIR}/hello_world.cpp from ${CMAKE_CURRENT_SOURCE_DIR}/hello_world.py"
                   COMMAND Python::Interpreter -m cython -3 --cplus
                           "${CMAKE_CURRENT_SOURCE_DIR}/hello_world_cython.pyx"
                           --output-file hello_world_cython.cpp
                   DEPENDS hello_world_cython.pyx
                   VERBATIM)

python_add_library(hello_world_cython MODULE hello_world_cython.cpp WITH_SOABI)

target_link_libraries(hello_world_cython PUBLIC legate::legate hello_world)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(platform_rpath_origin "\$ORIGIN")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(platform_rpath_origin "@loader_path")
else()
  message(FATAL_ERROR "Unsupported system: ${CMAKE_SYSTEM_NAME}, don't know how to set rpath 'origin' on this platform"
  )
endif()

set_property(TARGET hello_world_cython APPEND
             PROPERTY INSTALL_RPATH "${platform_rpath_origin}"
                      "${platform_rpath_origin}/../../")

install(TARGETS hello_world DESTINATION .)
install(TARGETS hello_world_cython DESTINATION .)
