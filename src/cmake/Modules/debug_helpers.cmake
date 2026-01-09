# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

if(NOT _LEGATE_CMAKE_PROPERTY_LIST)
  execute_process(COMMAND ${CMAKE_COMMAND} --help-property-list
                  OUTPUT_VARIABLE _LEGATE_CMAKE_PROPERTY_LIST)

  # Convert command output into a CMake list
  string(REGEX REPLACE ";" "\\\\;" _LEGATE_CMAKE_PROPERTY_LIST
                       "${_LEGATE_CMAKE_PROPERTY_LIST}")
  string(REGEX REPLACE "\n" ";" _LEGATE_CMAKE_PROPERTY_LIST
                       "${_LEGATE_CMAKE_PROPERTY_LIST}")
  list(REMOVE_DUPLICATES _LEGATE_CMAKE_PROPERTY_LIST)
endif()

function(print_all_properties)
  message(STATUS "CMAKE_PROPERTY_LIST = ${_LEGATE_CMAKE_PROPERTY_LIST}")
endfunction()

function(print_target_properties)
  set(options SHOW_UNSET)
  set(one_value_args TARGET)
  set(multi_value_keywords)
  cmake_parse_arguments(_LEGATE "${options}" "${one_value_args}"
                        "${multi_value_keywords}" ${ARGN})

  if(NOT TARGET ${_LEGATE_TARGET})
    message(FATAL_ERROR "There is no target named '${_LEGATE_TARGET}'")
  endif()

  foreach(property ${_LEGATE_CMAKE_PROPERTY_LIST})
    string(REPLACE "<CONFIG>" "${CMAKE_BUILD_TYPE}" property ${property})
    get_property(was_set TARGET ${_LEGATE_TARGET} PROPERTY ${property} SET)
    if(was_set OR _LEGATE_SHOW_UNSET)
      get_target_property(value ${_LEGATE_TARGET} ${property})
      message(STATUS "${_LEGATE_TARGET} ${property} = ${value}")
    endif()
  endforeach()
endfunction()
