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

# If legate was built with CUDA support, enable the CUDA language so the GPU task
# variants in saxpy.cu can be compiled. This must happen before the target is
# created so that its CUDA_ARCHITECTURES property is initialized correctly.
if(legate_USE_CUDA)
  # CUDA architectures must be populated before enable_language(CUDA), since that
  # call fixes the default. "all-major" mirrors legate's own default.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES "all-major")
  endif()

  enable_language(CUDA)
endif()

add_executable(saxpy saxpy.cc)

# saxpy.cc / saxpy.cu include the example's own header as <saxpy.h>, so the
# example source directory must be on the include path.
target_include_directories(saxpy PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

# Compile the GPU task variants alongside the CPU sources.
if(legate_USE_CUDA)
  target_sources(saxpy PRIVATE saxpy.cu)

  # saxpy.cu pulls in Thrust/CUB headers. Link against the same CCCL that legate
  # was built against (find_package(legate) above already located it and created
  # the CCCL::CCCL target) so that Thrust, CUB, and libcudacxx all come from one
  # consistent CCCL version. Otherwise Thrust/CUB resolve to the CUDA toolkit's
  # CCCL while libcudacxx is taken from legate's bundled CCCL; mixing versions
  # breaks compilation with preprocessor errors such as
  #   cuda/__iterator/counting_iterator.h: error:
  #     missing binary operator before token "("   (at `#if _CCCL_HAS_CONCEPTS()`)
  if(NOT TARGET CCCL::CCCL)
    find_package(CCCL REQUIRED)
  endif()
  target_link_libraries(saxpy PRIVATE CCCL::CCCL)
endif()

legate_configure_target(saxpy)
