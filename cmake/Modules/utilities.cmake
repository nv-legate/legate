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

macro(legate_string_escape_re_chars output_var input_var)
  # Escapes all special regex characters detailed at
  # https://cmake.org/cmake/help/latest/command/string.html#regex-specification
  string(REGEX REPLACE [[(\.|\-|\+|\*|\^|\$|\?|\||\(|\)|\[|\])]] [[\\\1]] ${output_var}
                       "${${input_var}}")
endmacro()

macro(legate_list_escape_re_chars output_var input_var)
  set(${output_var} ${${input_var}})
  list(TRANSFORM ${output_var} REPLACE [[(\.|\-|\+|\*|\^|\$|\?|\||\(|\)|\[|\])]] [[\\\1]])
endmacro()

function(legate_string_ends_with)
  set(options)
  set(one_value_args SRC ENDS_WITH RESULT_VAR)
  set(multi_value_args)

  cmake_parse_arguments(_LEGATE "${options}" "${one_value_args}" "${multi_value_args}"
                        ${ARGN})

  if(NOT _LEGATE_SRC)
    message(FATAL_ERROR "Must pass SRC")
  endif()

  if(NOT _LEGATE_ENDS_WITH)
    message(FATAL_ERROR "Must pass ENDS_WITH")
  endif()

  if(NOT _LEGATE_RESULT_VAR)
    message(FATAL_ERROR "Must pass RESULT_VAR")
  endif()

  string(STRIP "${_LEGATE_SRC}" src)
  string(STRIP "${_LEGATE_ENDS_WITH}" ends_with)

  legate_string_escape_re_chars(ends_with ends_with)

  if("${src}" MATCHES "${ends_with}$")
    set(${_LEGATE_RESULT_VAR} TRUE PARENT_SCOPE)
  else()
    set(${_LEGATE_RESULT_VAR} FALSE PARENT_SCOPE)
  endif()
endfunction()

function(legate_find_program VARIABLE_NAME PROGRAM_NAME)
  list(APPEND CMAKE_MESSAGE_CONTEXT "find_program")

  set(options)
  set(one_value_args)
  set(multi_value_args FIND_PROGRAM_ARGS)

  cmake_parse_arguments(_LEGATE "${options}" "${one_value_args}" "${multi_value_args}"
                        ${ARGN})

  message(CHECK_START "Searching for ${PROGRAM_NAME}")

  if(${VARIABLE_NAME})
    message(CHECK_PASS "using pre-found: ${${VARIABLE_NAME}}")
    return()
  endif()

  find_program(${VARIABLE_NAME} ${PROGRAM_NAME} ${_LEGATE_FIND_PROGRAM_ARGS})
  if(${VARIABLE_NAME})
    message(CHECK_PASS "found: ${${VARIABLE_NAME}}")
  else()
    message(CHECK_FAIL "not found")
  endif()
endfunction()
