#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

function(legate_generate_install_info_py)
  list(APPEND CMAKE_MESSAGE_CONTEXT "generate_install_info_py")
  # Legion sets this to "OFF" if not enabled, normalize it to an empty list instead
  if(NOT Legion_NETWORKS)
    set(Legion_NETWORKS "")
  endif()

  # Set by the pip wheels if they are being built, set to OFF if not
  if(NOT LEGATE_BUILD_PIP_WHEELS)
    set(LEGATE_BUILD_PIP_WHEELS OFF)
  endif()

  # CMake renders spaces in substitutions as "\ " which is a python syntax error. However,
  # converting the value to a CMake list (using semicolons) renders in substitutions as a
  # space-delimited string, which is what is desired. It is unclear if this CMake behavior
  # is documented/intendeed, but it it unlikely to change. Other approaches, e.g. VERBATIM
  # for the custom target, fail for other reasons.
  string(REPLACE " " ";" LEGATE_CONFIGURE_OPTIONS "${LEGATE_CONFIGURE_OPTIONS}")

  set(legate_LIB_NAME
      "$<TARGET_FILE_PREFIX:legate::legate>$<TARGET_FILE_BASE_NAME:legate::legate>")
  set(legate_FULL_LIB_NAME "$<TARGET_FILE_NAME:legate::legate>")

  cmake_path(SET install_info_tmp NORMALIZE
             "${CMAKE_CURRENT_BINARY_DIR}/install_info_tmp/install_info.py")
  # We need this 2-step because we make use of generator expressions, which
  # configure_file() does not support.
  #
  # The first configure_file() will emit the temporary file in our bin dir, replacing the
  # cmake values verbatim. It will contain generator expressions...
  configure_file("${LEGATE_CMAKE_DIR}/templates/install_info.py.in" "${install_info_tmp}"
                 @ONLY)
  # ...which this file(GENERATE) call will evaluate out. We could do this with a target,
  # that calls a cmake script that calls configure_file(), but it's much cleaner to do
  # this from the original generation stage.
  file(GENERATE OUTPUT "${LEGATE_DIR}/src/python/legate/install_info.py"
       INPUT "${install_info_tmp}")
endfunction()
