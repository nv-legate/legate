#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

include(CMakePushCheckState)
include(CheckCompilerFlag)

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

function(list_add_if_not_present list elem)
  list(FIND ${list} "${elem}" exists)
  if(exists EQUAL -1)
    list(APPEND ${list} "${elem}")
    set(${list} "${${list}}" PARENT_SCOPE)
  endif()
endfunction()

function(list_add_if_not_present_error list elem)
  list(FIND ${list} "${elem}" exists)
  if(exists EQUAL -1)
    list(APPEND ${list} "${elem}")
    set(${list} "${${list}}" PARENT_SCOPE)
  else()
    # present in list
    message(FATAL_ERROR "Element '${elem}' already present in list ${list}:" "${${list}}")
  endif()
endfunction()

macro(_legate_target_get_linked_libraries_in _target _outlist)
  list_add_if_not_present("${_outlist}" "${_target}")

  # get libraries
  get_target_property(target_type "${_target}" TYPE)
  if(${target_type} STREQUAL "INTERFACE_LIBRARY")
    get_target_property(libs "${_target}" INTERFACE_LINK_LIBRARIES)
  else()
    get_target_property(libs "${_target}" LINK_LIBRARIES)
  endif()

  foreach(lib IN LISTS libs)
    if(NOT TARGET "${lib}")
      continue()
    endif()

    list(FIND "${_outlist}" "${lib}" exists)
    if(NOT exists EQUAL -1)
      continue()
    endif()

    _legate_target_get_linked_libraries_in("${lib}" "${_outlist}")
  endforeach()
endmacro()

function(legate_target_get_target_dependencies)
  list(APPEND CMAKE_MESSAGE_CONTEXT "target_get_target_dependencies")

  set(options)
  set(one_value_args TARGET RESULT_VAR)
  set(multi_value_args)

  cmake_parse_arguments(_LEGATE "${options}" "${one_value_args}" "${multi_value_args}"
                        ${ARGN})

  foreach(var IN LISTS one_value_args)
    if(NOT _LEGATE_${var})
      message(FATAL_ERROR "Must pass ${var}")
    endif()
  endforeach()

  if(NOT TARGET "${_LEGATE_TARGET}")
    message(FATAL_ERROR "Target ${_LEGATE_TARGET} not a valid target")
  endif()

  set(out_list "")
  _legate_target_get_linked_libraries_in("${_LEGATE_TARGET}" out_list)
  set(${_LEGATE_RESULT_VAR} "${out_list}" PARENT_SCOPE)
endfunction()

function(legate_add_target_compile_options TARGET_NAME OPTION_LANG VIS OPTION_NAME)
  if(NOT ("${${OPTION_NAME}}" MATCHES ".*;.*"))
    # Using this form of separate_arguments() makes sure that quotes are respected when
    # the list is formed. Otherwise stuff like
    #
    # "--compiler-options='-foo -bar -baz'"
    #
    # becomes
    #
    # --compiler-options="'-foo";"-bar";"-baz'"
    #
    # which is obviously not what we wanted
    separate_arguments(${OPTION_NAME} NATIVE_COMMAND "${${OPTION_NAME}}")
  endif()

  set(lang_flags "$<$<COMPILE_LANGUAGE:${OPTION_LANG}>:${${OPTION_NAME}}>")
  target_compile_options("${TARGET_NAME}" "${VIS}" "${lang_flags}")
  # This is a nifty hack. We want to expose otherwise private flags to "private" targets
  # (for example, the tests, or cython bindings), but not expose them to other downstream
  # users.
  #
  # To achieve this, we add the same private flags as INTERFACE, with the caveat that they
  # are only activated if the *linked* target has a special LEGATE_INTERNAL_TARGET
  # property.
  #
  # If it does, we add the flags, if not, this has no effect.
  get_property(internal_prop_defined TARGET NONE PROPERTY LEGATE_INTERNAL_TARGET DEFINED)
  if(NOT internal_prop_defined)
    message(FATAL_ERROR "LEGATE_INTERNAL_TARGET was not defined as a property yet. "
                        "Probably some kind of refactoring has taken place and may have "
                        "caused this property to not be defined where it should be. See "
                        "the corresponding define_property() call in "
                        "src/cpp/CMakeLists.txt for more info")
  endif()
  if("${VIS}" STREQUAL "PRIVATE")
    set(has_prop "$<BOOL:$<TARGET_PROPERTY:LEGATE_INTERNAL_TARGET>>")
    set(lang_flags_if_has_secret_prop "$<${has_prop}:${lang_flags}>")
    target_compile_options("${TARGET_NAME}"
                           INTERFACE "$<BUILD_INTERFACE:${lang_flags_if_has_secret_prop}>"
    )
  endif()
endfunction()

function(legate_add_target_link_options TARGET_NAME VIS OPTION_NAME)
  if(NOT ("${${OPTION_NAME}}" MATCHES ".*;.*"))
    separate_arguments(${OPTION_NAME} NATIVE_COMMAND "${${OPTION_NAME}}")
  endif()
  if(${OPTION_NAME})
    target_link_options("${TARGET_NAME}" "${VIS}" "${${OPTION_NAME}}")
  endif()
endfunction()

function(legate_check_compiler_flag lang flag success_var)
  string(MAKE_C_IDENTIFIER "${flag}" flag_sanitized)
  message(CHECK_START "${lang} compiler supports ${flag}")

  cmake_push_check_state()
  set(CMAKE_REQUIRED_QUIET ON)
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -Werror")
  check_compiler_flag(${lang} "${flag}" ${flag_sanitized}_supported)
  cmake_pop_check_state()

  if(${flag_sanitized}_supported)
    message(CHECK_PASS "yes")
  else()
    message(CHECK_FAIL "no")
  endif()

  set(${success_var} "${${flag_sanitized}_supported}" PARENT_SCOPE)
endfunction()
