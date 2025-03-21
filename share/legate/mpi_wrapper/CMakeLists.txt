#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)

list(APPEND CMAKE_MESSAGE_CONTEXT "mpi_wrapper")

project(legate_mpi_wrapper VERSION 1.0 LANGUAGES C)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

set(BUILD_SHARED_LIBS ON)

# On by default because if this is being built by the user, and they install the wrapper,
# they will likely want to have a proper cmake installation so that their downstream
# projects can find this.
option(LEGATE_MPI_WRAPPER_SRC_INSTALL_RULES
       "Whether to generate install rules for source files" ON)

message(STATUS "Using build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Building shared library: ${BUILD_SHARED_LIBS}")
message(STATUS "Generating src install rules: ${LEGATE_MPI_WRAPPER_SRC_INSTALL_RULES}")

# ==========================================================================================
# main library

include(GNUInstallDirs)

find_package(MPI REQUIRED COMPONENTS C)

add_library(mpi_wrapper src/legate_mpi_wrapper/mpi_wrapper.c)
add_library(legate::mpi_wrapper ALIAS mpi_wrapper)

target_include_directories(mpi_wrapper
                           PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>)

if(LEGATE_MPI_WRAPPER_SRC_INSTALL_RULES)
  # Only generate the interface include paths if we are installing the headers.
  target_include_directories(mpi_wrapper
                             INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/legate_mpi_wrapper>
  )
endif()

target_link_libraries(mpi_wrapper PUBLIC MPI::MPI_C)

# Set a good default and make this something that can be modified when building the
# wrappers.
set(LEGATE_MPI_WRAPPER_LIB_NAME "legate_mpi_wrapper" CACHE STRING "Name of the library")

set_target_properties(mpi_wrapper
                      PROPERTIES LIBRARY_OUTPUT_NAME "${LEGATE_MPI_WRAPPER_LIB_NAME}"
                                 ARCHIVE_OUTPUT_NAME "${LEGATE_MPI_WRAPPER_LIB_NAME}"
                                 POSITION_INDEPENDENT_CODE ON
                                 LIBRARY_OUTPUT_DIRECTORY "${CMAKE_INSTALL_LIBDIR}"
                                 ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_INSTALL_LIBDIR}")

# ==========================================================================================
# Install rules
install(TARGETS mpi_wrapper DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        EXPORT mpi-wrapper-exports)

if(LEGATE_MPI_WRAPPER_SRC_INSTALL_RULES)
  install(FILES src/legate_mpi_wrapper/mpi_wrapper.h
                src/legate_mpi_wrapper/mpi_wrapper_types.h
          DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/legate_mpi_wrapper/legate_mpi_wrapper")

  set(cmake_install_base_dir "${CMAKE_INSTALL_LIBDIR}/cmake/legate_mpi_wrapper")
  set(config_output "${CMAKE_CURRENT_BINARY_DIR}/mpi_wrapperConfig.cmake")
  set(config_version_output "${CMAKE_CURRENT_BINARY_DIR}/mpi_wrapperConfigVersion.cmake")

  install(EXPORT mpi-wrapper-exports FILE mpi_wrapperTargets.cmake NAMESPACE legate::
          DESTINATION "${cmake_install_base_dir}")

  export(TARGETS mpi_wrapper NAMESPACE legate::
         FILE "${CMAKE_CURRENT_BINARY_DIR}/mpi_wrapperTargets.cmake")

  include(CMakePackageConfigHelpers)

  configure_package_config_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in"
                                "${config_output}"
                                INSTALL_DESTINATION "${cmake_install_base_dir}")

  write_basic_package_version_file("${config_version_output}"
                                   COMPATIBILITY SameMajorVersion ARCH_INDEPENDENT)

  install(FILES "${config_output}" "${config_version_output}"
          DESTINATION "${cmake_install_base_dir}")
endif()
