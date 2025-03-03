#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

function(legate_export_aedifix_post_config)
  list(APPEND CMAKE_MESSAGE_CONTEXT "export_aedifix_post_config")

  if(NOT AEDIFIX)
    message(VERBOSE "AEDIFIX not defined, no need to emit export config")
    return()
  endif()

  set(data "{}")
  foreach(var IN LISTS AEDIFIX_EXPORT_VARIABLES)
    string(JSON data SET "${data}" "${var}" "\"${${var}}\"")
  endforeach()
  message(VERBOSE "Wrote aedifix export config to ${AEDIFIX_EXPORT_CONFIG_PATH}")
  file(WRITE "${AEDIFIX_EXPORT_CONFIG_PATH}" "${data}")
endfunction()
