#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

if(legate_USE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES "all-major")
  endif()

  enable_language(CUDA)
endif()

add_executable(slice slice.cc)

target_include_directories(slice PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

# Compile the GPU task variants alongside the CPU sources.
if(legate_USE_CUDA)
  target_sources(slice PRIVATE slice.cu)

  if(NOT TARGET CCCL::CCCL)
    find_package(CCCL REQUIRED)
  endif()
  target_link_libraries(slice PRIVATE CCCL::CCCL)
endif()

legate_configure_target(slice)
