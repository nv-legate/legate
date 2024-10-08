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

include("${LEGATE_CMAKE_DIR}/Modules/utilities.cmake")

function(_legate_install_debug_syms_macos target install_dir)
  list(APPEND CMAKE_MESSAGE_CONTEXT "macos")

  get_target_property(imported "${target}" IMPORTED)
  if(imported)
    # Nothing to do for imported targets
    message(VERBOSE "cannot install debug information for ${target}"
            "(target was imported, we did not generate it)")
    return()
  endif()
  get_target_property(target_type "${target}" TYPE)
  if((target_type STREQUAL "SHARED_LIBRARY") OR (target_type STREQUAL "EXECUTABLE"))
    # We want to install the dsymutil stuff directly next to the installed binary/lib.
    # Unfortunately, there is no way to query where that path is (CMake simply doesn't
    # keep track). We could write it in ourselves by setting a custom property, but then
    # people will forget to do that and this breaks.
    #
    # So instead we need them to tell us exactly where to put it...
    install(DIRECTORY "$<TARGET_FILE:${target}>.dSYM" DESTINATION "${install_dir}")
  endif()
endfunction()

function(legate_install_debug_symbols)
  list(APPEND CMAKE_MESSAGE_CONTEXT "debug_symbols")

  set(options RECURSIVE)
  set(one_value_args TARGET INSTALL_DIR)
  set(multi_value_keywords)
  cmake_parse_arguments(_LEGATE "${options}" "${one_value_args}"
                        "${multi_value_keywords}" ${ARGN})

  foreach(var TARGET INSTALL_DIR)
    if(NOT _LEGATE_${var})
      message(FATAL_ERROR "Must pass ${var}")
    endif()
  endforeach()

  if(NOT TARGET "${_LEGATE_TARGET}")
    message(FATAL_ERROR "Target '${_LEGATE_TARGET}' is not a target")
  endif()

  get_property(debug_hooks_installed GLOBAL PROPERTY LEGATE_INSTALLED_DEBUG_SYMBOL_HOOKS)
  if(NOT debug_hooks_installed)
    message(FATAL_ERROR "Must call legate_configure_debug_symbols() first")
  endif()

  if(_LEGATE_RECURSIVE)
    legate_target_get_target_dependencies(TARGET "${_LEGATE_TARGET}"
                                          RESULT_VAR target_list)
  else()
    set(target_list "${_LEGATE_TARGET}")
  endif()

  if((CMAKE_BUILD_TYPE STREQUAL "Debug") OR (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo"))
    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
      foreach(target IN LISTS target_list)
        _legate_install_debug_syms_macos(${target} ${_LEGATE_INSTALL_DIR})
      endforeach()
    endif()
    # nothing to do for other OS's for now
  endif()
endfunction()

# ------------------------------------------------------------------------------------------

macro(_legate_configure_debug_symbols_macos)
  list(APPEND CMAKE_MESSAGE_CONTEXT "macos")

  legate_find_program(LEGATE_DSYMUTIL dsymutil FIND_PROGRAM_ARGS REQUIRED)

  get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
  list(REMOVE_ITEM languages NONE)

  foreach(lang IN LISTS languages)
    foreach(var LINK_EXECUTABLE CREATE_SHARED_LIBRARY)
      list(APPEND CMAKE_${lang}_${var} "${LEGATE_DSYMUTIL} <TARGET>")
      set(CMAKE_${lang}_${var} "${CMAKE_${lang}_${var}}" PARENT_SCOPE)
      message(STATUS "Installed dsymutil hooks for CMAKE_${lang}_${var}")
    endforeach()
  endforeach()

  list(POP_BACK CMAKE_MESSAGE_CONTEXT)
endmacro()

function(legate_configure_debug_symbols)
  list(APPEND CMAKE_MESSAGE_CONTEXT "configure_debug_symbols")

  if((CMAKE_BUILD_TYPE STREQUAL "Debug") OR (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo"))
    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
      _legate_configure_debug_symbols_macos()
    endif()
    # nothing to do for other OS's for now
  endif()

  set_property(GLOBAL PROPERTY LEGATE_INSTALLED_DEBUG_SYMBOL_HOOKS ON)
endfunction()
