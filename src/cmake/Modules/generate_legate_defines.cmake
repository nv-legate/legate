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

include_guard(GLOBAL)

include(GNUInstallDirs)

function(legate_generate_legate_defines)
  list(APPEND CMAKE_MESSAGE_CONTEXT "generate_legate_defines")

  # Must set these all to exactly 1 so that the configure file is correctly generated
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(LEGATE_USE_DEBUG 1)
  endif()

  if(Legion_USE_CUDA)
    set(LEGATE_USE_CUDA 1)
  endif()

  if(Legion_NETWORKS)
    set(LEGATE_USE_NETWORK 1)
  endif()

  if(Legion_USE_OpenMP)
    set(LEGATE_USE_OPENMP 1)
  endif()

  if(Legion_USE_CUDA AND CAL_DIR)
    set(LEGATE_USE_CAL 1)
  endif()

  if(NOT LEGATE_CONFIGURE_OPTIONS)
    set(LEGATE_CONFIGURE_OPTIONS "<unknown configure options>")
  endif()

  configure_file(${LEGATE_CMAKE_DIR}/templates/legate_defines.h.in
                 "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/legate/legate_defines.h"
                 @ONLY)
endfunction()
