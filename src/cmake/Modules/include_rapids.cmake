#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

function(_legate_download_rapids DEST_PATH)
  set(expected_hash "")
  if(NOT rapids-cmake-version)
    # default
    set(rapids-cmake-version 24.12)
    set(rapids-cmake-sha "4cb2123dc08ef5d47ecdc9cc51c96bea7b5bb79c")
    # This hash needs to be manually updated every time we bump rapids-cmake
    set(expected_hash
        EXPECTED_HASH
        SHA256=1f4575699380b7bbf0a3363970ad83fdf0778f56a6ac36d7931fdf154d336448)
  endif()

  set(file_name
      "https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-${rapids-cmake-version}/RAPIDS.cmake"
  )
  file(DOWNLOAD "${file_name}" "${DEST_PATH}" ${expected_hash} STATUS status)

  list(GET status 0 code)
  if(NOT code EQUAL 0)
    list(GET status 1 reason)
    message(FATAL_ERROR "Error (${code}) when downloading ${file_name}: ${reason}")
  endif()
endfunction()

macro(legate_include_rapids)
  list(APPEND CMAKE_MESSAGE_CONTEXT "include_rapids")

  if(NOT _LEGATE_HAS_RAPIDS)
    set(legate_rapids_file "${CMAKE_CURRENT_BINARY_DIR}/LEGATE_RAPIDS.cmake")

    if(NOT EXISTS ${legate_rapids_file})
      _legate_download_rapids("${legate_rapids_file}")
    endif()
    include("${legate_rapids_file}")

    unset(legate_rapids_file)
    set(_LEGATE_HAS_RAPIDS ON)
  endif()
  list(POP_BACK CMAKE_MESSAGE_CONTEXT)
endmacro()
