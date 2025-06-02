#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

function(_legate_download_rapids DEST_PATH)
  set(expected_hash "")
  if(rapids-cmake-version)
    # The way the current code is structured, we rely on a few things from file(DOWNLOAD).
    #
    # 1. If the file does not exist, file(DOWNLOAD) will download it for us.
    # 2. If the file exists, but the hash doesn't match, file(DOWNLOAD) will re-download it
    #    for us.
    #
    # So if the user is setting the rapids-cmake-version, then we don't have the file
    # hash, which means that file(DOWNLOAD) might see some stale version of the file and
    # conclude it has nothing to do (because how could it know that it should re-download
    # the file). So we delete the existing file just to be safe.
    file(REMOVE "${DEST_PATH}")
  else()
    # default
    set(rapids-cmake-version "25.04")
    set(rapids-cmake-sha "757f9fddd00ef80c26bde4e6ecd4edcd0c7c68f5")

    # These need to be seen by the include(legate_rapids_file) call
    set(rapids-cmake-version "${rapids-cmake-version}" PARENT_SCOPE)
    set(rapids-cmake-sha "${rapids-cmake-sha}" PARENT_SCOPE)

    # This hash needs to be manually updated every time we bump rapids-cmake
    set(expected_hash
        EXPECTED_HASH
        SHA256=8044374f0460a09792894f8390244b896a4c0ede9c8b70b0c7caa522c40a45e9)
  endif()

  set(file_name
      "https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-${rapids-cmake-version}/RAPIDS.cmake"
  )
  # Retry the download up to 5 times (foreach() is inclusive). Also cmake-lint seemingly
  # doesn't know about range-foreach...
  foreach(idx RANGE 1 5) # cmake-lint: disable=E1120
    file(DOWNLOAD "${file_name}" "${DEST_PATH}" ${expected_hash} STATUS status)

    list(GET status 0 code)
    if(code EQUAL 0)
      return()
    endif()
    message(VERBOSE "Failed to download ${file_name}, retrying.")
    # CMake has no builtin sleep command, use idx to implement a back-off
    execute_process(COMMAND ${CMAKE_COMMAND} -E sleep "${idx}")
  endforeach()

  # If we reach this point, we have failed to download the file.
  #
  # If there is an error, either when downloading the file, or computing the checksum,
  # CMake will not delete the file, it will simply leave it there. Suppose for example
  # that the cause of the problem is a transient network problem. In this case the user
  # will just re-run configure (to try again), but the file existence check below will see
  # a file and not attempt to re-download it.
  #
  # So we need to delete it ourselves.
  file(REMOVE "${DEST_PATH}")
  list(GET status 1 reason)
  message(FATAL_ERROR "Error (${code}) when downloading ${file_name}: ${reason}")
endfunction()

macro(legate_include_rapids)
  list(APPEND CMAKE_MESSAGE_CONTEXT "include_rapids")

  if(NOT _LEGATE_HAS_RAPIDS)
    set(legate_rapids_file "${CMAKE_CURRENT_BINARY_DIR}/LEGATE_RAPIDS.cmake")

    _legate_download_rapids("${legate_rapids_file}")
    include("${legate_rapids_file}")

    unset(legate_rapids_file)
    set(_LEGATE_HAS_RAPIDS ON)
  endif()
  list(POP_BACK CMAKE_MESSAGE_CONTEXT)
endmacro()
