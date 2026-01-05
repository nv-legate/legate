#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

# See https://discourse.cmake.org/t/escaping-strings-for-output-to-json-file/5297
function(_legate_json_esc ret_var val_str)
  string(JSON tmp_json SET "{}" "${val_str}" "0")
  string(REGEX REPLACE "^\\{[ \t\r\n]*" "" tmp_json "${tmp_json}")
  string(REGEX REPLACE "[ \t\r\n]*:[ \t\r\n]*0[ \t\r\n]*\\}$" "" tmp_json "${tmp_json}")
  if(NOT "${tmp_json}" MATCHES "^\"[^\n]*\"")
    message(FATAL_ERROR "Internal error: unexpected output: '${tmp_json}'")
  endif()
  set(${ret_var} "${tmp_json}" PARENT_SCOPE)
endfunction()

function(legate_export_aedifix_post_config)
  list(APPEND CMAKE_MESSAGE_CONTEXT "export_aedifix_post_config")

  if(NOT AEDIFIX)
    message(VERBOSE "AEDIFIX not defined, no need to emit export config")
    return()
  endif()

  set(data "{}")
  foreach(var IN LISTS AEDIFIX_EXPORT_VARIABLES)
    _legate_json_esc(value "${${var}}")
    string(JSON data SET "${data}" "${var}" "${value}")
  endforeach()
  message(VERBOSE "Wrote aedifix export config to ${AEDIFIX_EXPORT_CONFIG_PATH}")
  file(WRITE "${AEDIFIX_EXPORT_CONFIG_PATH}" "${data}")
endfunction()
