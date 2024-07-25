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

macro(legate_core_string_escape_re_chars output_var input_var)
  # Escapes all special regex characters detailed at
  # https://cmake.org/cmake/help/latest/command/string.html#regex-specification
  string(REGEX REPLACE [[(\.|\-|\+|\*|\^|\$|\?|\||\(|\)|\[|\])]] [[\\\1]] ${output_var}
                       "${${input_var}}")
endmacro()

macro(legate_core_list_escape_re_chars output_var input_var)
  set(${output_var} ${${input_var}})
  list(TRANSFORM ${output_var} REPLACE [[(\.|\-|\+|\*|\^|\$|\?|\||\(|\)|\[|\])]] [[\\\1]])
endmacro()

function(legate_core_string_ends_with)
  set(options)
  set(one_value_args SRC ENDS_WITH RESULT_VAR)
  set(multi_value_args)

  cmake_parse_arguments(_LEGATE_CORE "${options}" "${one_value_args}"
                        "${multi_value_args}" ${ARGN})

  if(NOT_LEGATE_CORE_SRC)
    message(FATAL_ERROR "Must pass SRC")
  endif()

  if(NOT_LEGATE_CORE_ENDS_WITH)
    message(FATAL_ERROR "Must pass ENDS_WITH")
  endif()

  if(NOT_LEGATE_CORE_RESULT_VAR)
    message(FATAL_ERROR "Must pass RESULT_VAR")
  endif()

  string(STRIP "${_LEGATE_CORE_SRC}" src)
  string(STRIP "${_LEGATE_CORE_ENDS_WITH}" ends_with)

  legate_core_string_escape_re_chars(ends_with ends_with)

  message("src: ${src}")
  message("ends_with: ${ends_with}")

  if("${src}" MATCHES "${ends_with}$")
    set(${_LEGATE_CORE_RESULT_VAR} TRUE PARENT_SCOPE)
  else()
    set(${_LEGATE_CORE_RESULT_VAR} FALSE PARENT_SCOPE)
  endif()
endfunction()
