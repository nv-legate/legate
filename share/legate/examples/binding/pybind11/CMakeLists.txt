#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

cmake_minimum_required(VERSION 3.17)

# Scikit-build-core sets these values for you, or you can just hard-code the name and
# version.
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_SHARED_LIBS ON)

find_package(legate QUIET)

if(NOT legate_FOUND)
  if(DEFINED ENV{LEGATE_DIR} AND DEFINED ENV{LEGATE_ARCH})
    set(legate_DIR "$ENV{LEGATE_DIR}/$ENV{LEGATE_ARCH}/cmake_build")
  endif()
  find_package(legate REQUIRED QUIET)
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Add a library using FindPython's tooling (pybind11 also provides a helper like this)
python_add_library(hello_world_pybind11 MODULE hello_world.cc WITH_SOABI)
target_link_libraries(hello_world_pybind11 PUBLIC legate::legate
                      PRIVATE pybind11::headers)

# This is passing in the version as a define just as an example
target_compile_definitions(hello_world_pybind11 PRIVATE VERSION_INFO=${PROJECT_VERSION})

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(platform_rpath_origin "\$ORIGIN")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(platform_rpath_origin "@loader_path")
else()
  message(FATAL_ERROR "Unsupported system: ${CMAKE_SYSTEM_NAME}, don't know how to set rpath 'origin' on this platform"
  )
endif()

set_property(TARGET hello_world_pybind11 APPEND
             PROPERTY INSTALL_RPATH "${platform_rpath_origin}/../../")

# The install directory is the output (wheel) directory
install(TARGETS hello_world_pybind11 DESTINATION .)
