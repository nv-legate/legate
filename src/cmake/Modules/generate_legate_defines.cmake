#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

function(legate_generate_legate_defines)
  list(APPEND CMAKE_MESSAGE_CONTEXT "generate_legate_defines")

  # Must set these all to exactly 1 so that the configure file is correctly generated
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(LEGATE_USE_DEBUG 1)
  endif()

  if(legate_USE_CUDA)
    set(LEGATE_USE_CUDA 1)
  endif()

  if(legate_USE_GASNET OR legate_USE_UCX OR legate_USE_MPI)
    set(LEGATE_USE_NETWORK 1)
  endif()

  if(Legion_USE_OpenMP)
    set(LEGATE_USE_OPENMP 1)
  endif()

  if(legate_USE_HDF5)
    set(LEGATE_USE_HDF5 1)
  endif()

  if(legate_USE_HDF5_VFD_GDS)
    set(LEGATE_USE_HDF5_VFD_GDS 1)
  endif()

  if(legate_USE_NCCL)
    set(LEGATE_USE_NCCL 1)
  endif()

  if(legate_USE_UCX)
    set(LEGATE_USE_UCX 1)
  endif()

  if(legate_USE_MPI OR legate_USE_GASNET)
    set(LEGATE_USE_MPI 1)
  endif()

  if(NOT LEGATE_CONFIGURE_OPTIONS)
    set(LEGATE_CONFIGURE_OPTIONS "<unknown configure options>")
  endif()

  configure_file("${LEGATE_CMAKE_DIR}/templates/legate_defines.h.in"
                 "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/legate/legate_defines.h"
                 @ONLY)
endfunction()
