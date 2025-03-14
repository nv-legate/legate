#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)

list(APPEND CMAKE_MESSAGE_CONTEXT "realm_ucp_bootstrap")

project(realm_ucp_bootstrap VERSION 1.0 LANGUAGES C)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

set(BUILD_SHARED_LIBS ON)

# On by default because if this is being built by the user, and they install the wrapper,
# they will likely want to have a proper cmake installation so that their downstream
# projects can find this.
option(LEGATE_MPI_WRAPPER_SRC_INSTALL_RULES
       "Whether to generate install rules for source files" OFF)

message(STATUS "Using build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Building shared library: ${BUILD_SHARED_LIBS}")
message(STATUS "Generating src install rules: ${LEGATE_MPI_WRAPPER_SRC_INSTALL_RULES}")

# ==========================================================================================
# main library

include(GNUInstallDirs)

find_package(MPI REQUIRED COMPONENTS C)

add_library(realm_ucp_bootstrap
            bootstrap_mpi.c
            bootstrap.h
            bootstrap_util.h)

target_include_directories(realm_ucp_bootstrap
                           PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>)

if(LEGATE_MPI_WRAPPER_SRC_INSTALL_RULES)
  # Only generate the interface include paths if we are installing the headers.
  target_include_directories(realm_ucp_bootstrap
                             INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/realm_ucp_bootstrap>
  )
endif()

target_link_libraries(realm_ucp_bootstrap PUBLIC MPI::MPI_C)

set(REALM_UCP_BOOTSTRAP_LIB_NAME "realm_ucp_bootstrap_mpi" CACHE STRING "Name of the library")

set_target_properties(realm_ucp_bootstrap
                      PROPERTIES LIBRARY_OUTPUT_NAME "${REALM_UCP_BOOTSTRAP_LIB_NAME}"
                                 ARCHIVE_OUTPUT_NAME "${REALM_UCP_BOOTSTRAP_LIB_NAME}"
                                 POSITION_INDEPENDENT_CODE ON
                                 LIBRARY_OUTPUT_DIRECTORY "${CMAKE_INSTALL_LIBDIR}"
                                 ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_INSTALL_LIBDIR}")

# ==========================================================================================
# Install rules
install(TARGETS realm_ucp_bootstrap DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        EXPORT realm-ucp-bootstrap-exports)

if(LEGATE_MPI_WRAPPER_SRC_INSTALL_RULES)
  install(FILES bootstrap.h
                bootstrap_util.h
          DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/realm_ucp_bootstrap/realm_ucp_bootstrap")

  set(cmake_install_base_dir "${CMAKE_INSTALL_LIBDIR}/cmake/realm_ucp_bootstrap")
  set(config_output "${CMAKE_CURRENT_BINARY_DIR}/realm_ucp_bootstrapConfig.cmake")
  set(config_version_output "${CMAKE_CURRENT_BINARY_DIR}/realm_ucp_bootstrapConfigVersion.cmake")

  install(EXPORT realm-ucp-bootstrap-exports FILE realm_ucp_bootstrapTargets.cmake NAMESPACE legate::
          DESTINATION "${cmake_install_base_dir}")

  export(TARGETS realm_ucp_bootstrap NAMESPACE legate::
         FILE "${CMAKE_CURRENT_BINARY_DIR}/realm_ucp_bootstrapTargets.cmake")

  include(CMakePackageConfigHelpers)

  configure_package_config_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in"
                                "${config_output}"
                                INSTALL_DESTINATION "${cmake_install_base_dir}")

  write_basic_package_version_file("${config_version_output}"
                                   COMPATIBILITY SameMajorVersion ARCH_INDEPENDENT)

  install(FILES "${config_output}" "${config_version_output}"
          DESTINATION "${cmake_install_base_dir}")
endif()
