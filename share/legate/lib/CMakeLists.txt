#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#=============================================================================

cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)

if(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

if(NOT DEFINED BUILD_SHARED_LIBS)
  set(BUILD_SHARED_LIBS ON)
endif()

project(legate_core_mpi_wrapper VERSION 1.0 LANGUAGES CXX)

# ==========================================================================================
# main library

find_package(MPI REQUIRED COMPONENTS CXX)

add_library(mpi_wrapper mpi_wrapper.cc)

target_include_directories(mpi_wrapper PRIVATE "${CMAKE_CURRENT_LIST_DIR}")

target_link_libraries(mpi_wrapper PRIVATE MPI::MPI_CXX)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(rpath_origin "\$ORIGIN")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(rpath_origin "@loader_path")
else()
  message(FATAL_ERROR "Unsupported system: ${CMAKE_SYSTEM_NAME}, don't know how to set rpath 'origin' on this platform"
  )
endif()

set(lib_name "lgcore_mpi_wrapper")

include(GNUInstallDirs)

set_target_properties(mpi_wrapper
                      PROPERTIES LIBRARY_OUTPUT_NAME "${lib_name}"
                                 ARCHIVE_OUTPUT_NAME "${lib_name}"
                                 POSITION_INDEPENDENT_CODE ON
                                 LIBRARY_OUTPUT_DIRECTORY "${CMAKE_INSTALL_LIBDIR}"
                                 ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_INSTALL_LIBDIR}"
                                 BUILD_RPATH "${rpath_origin}"
                                 INSTALL_RPATH "${rpath_origin}")

install(TARGETS mpi_wrapper DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        EXPORT mpi-wrapper-exports)

# ==========================================================================================
# test executable to ensure everything works

add_executable(wrapper_tester wrapper_test.cc)

target_link_libraries(wrapper_tester mpi_wrapper)

set_target_properties(wrapper_tester
                      PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_INSTALL_BINDIR}
                                 POSITION_INDEPENDENT_CODE ON
                                 BUILD_RPATH "${rpath_origin}"
                                 INSTALL_RPATH "${rpath_origin}")

add_custom_command(TARGET wrapper_tester
                   POST_BUILD
                   COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/wrapper_tester
                   COMMENT "Running wrapper check")
