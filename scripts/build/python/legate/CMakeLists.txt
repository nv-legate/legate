#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

cmake_minimum_required(VERSION 3.26.4)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../../../../VERSION" LEGATE_VERSION)
string(STRIP "${LEGATE_VERSION}" LEGATE_VERSION)
project(legate-python VERSION ${LEGATE_VERSION} LANGUAGES CXX)

set(CMAKE_CUDA_RUNTIME_LIBRARY "STATIC")
set(CMAKE_CUDA_ARCHITECTURES all-major)
set(BUILD_SHARED_LIBS ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_INSTALL_LIBDIR lib64)

# Make it clear we are building pip wheels.
set(LEGATE_BUILD_PIP_WHEELS ON)

# Set defaults for the wheels build.
if(UNIX AND NOT APPLE)
  set(legate_USE_CUDA ON)
  set(legate_USE_UCX OFF)
  set(legate_USE_NCCL OFF)
endif()

add_subdirectory(../../../../src legate-all)
