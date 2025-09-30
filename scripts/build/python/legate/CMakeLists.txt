#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  set(Legion_NETWORKS ucx)
  set(legate_USE_NCCL ON)
  # Disable CAL for now due to missing link in the wheel.
  set(legate_USE_CAL OFF)
endif()

add_subdirectory(../../../../src legate-all)

# RealmRuntime may or may not be an ALIAS target depending on whether the version used by
# Legion is the one before or after Realm split off from Legion.
set(REALM_TARGET RealmRuntime)
get_target_property(aliased_target "${REALM_TARGET}" ALIASED_TARGET)
if(aliased_target)
  set(REALM_TARGET "${aliased_target}")
endif()

# Add the necessary rpaths for the wheels build.
if(UNIX AND NOT APPLE)
  set(rpaths "$ORIGIN/../../nvidia/nccl/lib")
  set_property(TARGET legate PROPERTY INSTALL_RPATH ${rpaths} APPEND)
  set(rpaths "$ORIGIN/../../nvidia/libcal/cu12/lib")
  set_property(TARGET ${REALM_TARGET} PROPERTY INSTALL_RPATH ${rpaths} APPEND)
endif()

# Install the legate wrappers if they exist.
if(LEGATE_WRAPPER_DIR)
  install(DIRECTORY ${LEGATE_WRAPPER_DIR}/lib64/
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/legate/wrapper)
  set(rpaths "$ORIGIN/legate/wrapper")
  # Add the rpath to the legate and RealmRuntime targets so they can load the wrappers.
  set_property(TARGET legate PROPERTY INSTALL_RPATH ${rpaths} APPEND)
  set_property(TARGET ${REALM_TARGET} PROPERTY INSTALL_RPATH ${rpaths} APPEND)
endif()
