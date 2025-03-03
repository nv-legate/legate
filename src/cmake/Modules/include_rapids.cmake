#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

macro(legate_include_rapids)
  list(APPEND CMAKE_MESSAGE_CONTEXT "include_rapids")

  if(NOT _LEGATE_HAS_RAPIDS)
    if(NOT rapids-cmake-version)
      # default
      set(rapids-cmake-version 24.12)
      set(rapids-cmake-sha "4cb2123dc08ef5d47ecdc9cc51c96bea7b5bb79c")
    endif()

    if(NOT EXISTS ${CMAKE_BINARY_DIR}/LEGATE_RAPIDS.cmake)
      file(DOWNLOAD
           https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-${rapids-cmake-version}/RAPIDS.cmake
           ${CMAKE_BINARY_DIR}/LEGATE_RAPIDS.cmake)
    endif()
    include(${CMAKE_BINARY_DIR}/LEGATE_RAPIDS.cmake)
    set(_LEGATE_HAS_RAPIDS ON)
  endif()
  list(POP_BACK CMAKE_MESSAGE_CONTEXT)
endmacro()
