#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

cmake_minimum_required(VERSION 3.26.4)

project(legate-python VERSION 25.03.00 LANGUAGES CXX)

set(CMAKE_CUDA_RUNTIME_LIBRARY "STATIC")
set(CMAKE_CUDA_ARCHITECTURES all-major)
set(BUILD_SHARED_LIBS ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# Make it clear we are building pip wheels.
set(LEGATE_BUILD_PIP_WHEELS ON)

# Set defaults for the wheels build.
set(Legion_USE_CUDA ON)
set(Legion_NETWORKS ucx)
# Disable CAL for now due to missing link in the wheel.
set(legate_USE_CAL OFF)

add_subdirectory(../../../../src legate-all)

# Add the necessary rpaths for the wheels build.
set(rpaths "$ORIGIN/../../nvidia/nccl/lib")
set_property(TARGET legate PROPERTY INSTALL_RPATH ${rpaths} APPEND)
set(rpaths "$ORIGIN/../../nvidia/libcal/cu12/lib")
set_property(TARGET RealmRuntime PROPERTY INSTALL_RPATH ${rpaths} APPEND)

# Install the legate wrappers if they exist.
if(LEGATE_WRAPPER_DIR)
  install(DIRECTORY ${LEGATE_WRAPPER_DIR}/lib64/
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/legate/wrapper)
  set(rpaths "$ORIGIN/legate/wrapper")
  # Add the rpath to the legate and RealmRuntime targets so they can load the wrappers.
  set_property(TARGET legate PROPERTY INSTALL_RPATH ${rpaths} APPEND)
  set_property(TARGET RealmRuntime PROPERTY INSTALL_RPATH ${rpaths} APPEND)
endif()
