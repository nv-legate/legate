#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

cmake_minimum_required(VERSION 3.26.4)

project(legate_wrapper)

set(MPI_NAME_SUFFIX "")
if(LEGATE_WRAPPER_MPI_SUFFIX)
  set(MPI_NAME_SUFFIX "_${LEGATE_WRAPPER_MPI_SUFFIX}")
endif()

# We don't want to install the source for the wrappers
set(LEGATE_MPI_WRAPPER_SRC_INSTALL_RULES OFF)
set(LEGATE_MPI_WRAPPER_LIB_NAME "legate_mpi_wrapper${MPI_NAME_SUFFIX}")
add_subdirectory(../../../share/legate/mpi_wrapper
                 "${CMAKE_BINARY_DIR}/legate_mpi_wrapper")

set(REALM_UCP_BOOTSTRAP_LIB_NAME "realm_ucp_bootstrap_mpi${MPI_NAME_SUFFIX}")
add_subdirectory(../../../share/legate/realm_ucp_bootstrap
                 "${CMAKE_BINARY_DIR}/realm_ucp_bootstrap")
