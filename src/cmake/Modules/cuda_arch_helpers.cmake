#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

function(_legate_get_supported_arch_list_nvcc dest_var var_name)
  list(APPEND CMAKE_MESSAGE_CONTEXT "nvcc")

  if(legate_NVCC_SUPPORTED_ARCH_LIST)
    set(${dest_var} "${legate_NVCC_SUPPORTED_ARCH_LIST}" PARENT_SCOPE)
    return()
  endif()

  set(cmd "${CMAKE_CUDA_COMPILER}" "-arch-ls")
  execute_process(COMMAND ${cmd}
                  OUTPUT_VARIABLE arch_list
                  ERROR_VARIABLE err
                  RESULT_VARIABLE result
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(result OR err)
    message(FATAL_ERROR "Failed to auto-detect the list of supported CUDA "
                        "architectures from NVCC. Please set ${var_name} to "
                        "the appropriate list of architectures. Ran:\n"
                        "${cmd}\n"
                        "(${result}): ${err}")
  endif()

  string(REPLACE "\n" ";" arch_list ${arch_list})
  list(TRANSFORM arch_list REPLACE [=[(compute|arch)_]=] "")
  list(TRANSFORM arch_list STRIP)
  list(REMOVE_ITEM arch_list "")
  # Use natural comparison, so that 100 does not end up before 90 etc.
  list(SORT arch_list COMPARE NATURAL)

  set(legate_NVCC_SUPPORTED_ARCH_LIST "${arch_list}"
      CACHE INTERNAL "List of supported CUDA arch values")
  set(${dest_var} "${arch_list}" PARENT_SCOPE)
endfunction()

function(legate_set_default_cuda_arch)
  list(APPEND CMAKE_MESSAGE_CONTEXT "set_default_cuda_arch")

  set(options)
  set(oneValueArgs DEST_VAR)
  set(multiValueArgs)
  cmake_parse_arguments(_LEGATE "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  if(NOT _LEGATE_DEST_VAR)
    message(FATAL_ERROR "Must pass DEST_VAR")
  endif()

  if(${_LEGATE_DEST_VAR} STREQUAL "all-major")
    message(STATUS "arch variable ${_LEGATE_DEST_VAR}=${${_LEGATE_DEST_VAR}}, "
                   "translating to all supported major architectures")
    set(ONLY_MAJOR ON)
  elseif(${_LEGATE_DEST_VAR} STREQUAL "all")
    message(STATUS "arch variable ${_LEGATE_DEST_VAR}=${${_LEGATE_DEST_VAR}}, "
                   "translating to all supported architectures")
    set(ONLY_MAJOR OFF)
  elseif(DEFINED ${_LEGATE_DEST_VAR})
    # Variable was already set by user to something else, don't mess with it.  Use of
    # DEFINED is deliberate. We want to handle the case where DEST_VAR is "OFF" (which we
    # should leave as-is).
    message(STATUS "arch variable already pre-defined: "
                   "${_LEGATE_DEST_VAR}=${${_LEGATE_DEST_VAR}}")
    if(${_LEGATE_DEST_VAR} MATCHES [=[^[0-5][0-9]$]=])
      message(FATAL_ERROR "CUDA architecture ${${_LEGATE_DEST_VAR}} is not supported.")
    endif()

    return()
  else()
    message(STATUS "arch variable ${_LEGATE_DEST_VAR} undefined, default to all-major")
    set(ONLY_MAJOR ON)
  endif()

  # At this point we should be left with only 2 options:
  #
  # * all-major
  # * all
  #
  # Since the above if tree should have early-returned if the input variable had any other
  # value. Thus, if this variable is unset, then we know that we didn't cover a branch
  # above.
  if(NOT DEFINED ONLY_MAJOR)
    message(FATAL_ERROR "Bug in legate cmake, failed to set ONLY_MAJOR")
  endif()

  set(arch_list)
  # Currently only know how to handle NVCC, but if we ever support clang, then this is
  # where we'd do that
  _legate_get_supported_arch_list_nvcc(arch_list "${_LEGATE_DEST_VAR}")

  # Remove < sm_70 as that is the lowest version supported by the project.
  list(FILTER arch_list EXCLUDE REGEX [=[^[0-6][0-9]$]=])
  if(ONLY_MAJOR)
    # Remove any non-major architectures, they should all numeric-only except for Hopper,
    # which strangely has sm_90a.
    #
    # So given arch_list = [75, 90, 90a, 91, 91a, 100, 120] ...
    list(FILTER arch_list EXCLUDE REGEX [=[^[0-9]+[1-9][a-z]?$]=])
    # We now have
    #
    # arch_list = [90, 90a, 100, 120]
    list(FILTER arch_list EXCLUDE REGEX [=[^[0-9]+[a-z]$]=])
    # And now we have
    #
    # arch_list = [90, 100, 120]
    #
  endif()

  # A CMake architecture list entry of "80" means to build both compute and sm. What we
  # want is for the newest arch only to build that way, while the rest build only for sm.
  list(POP_BACK arch_list latest_arch)
  # The regex is there to match any numeric-only (and the odd 90a as discussed above)
  # architectures, and skip those already containing -real or -virtual.
  list(TRANSFORM arch_list APPEND "-real" REGEX [=[^[0-9]+[a-z]?$]=])
  list(APPEND arch_list ${latest_arch})

  set(${_LEGATE_DEST_VAR} "${arch_list}")
  set(${_LEGATE_DEST_VAR} "${${_LEGATE_DEST_VAR}}" PARENT_SCOPE)

  message(STATUS "Set default CUDA architectures: "
                 "${_LEGATE_DEST_VAR}=${${_LEGATE_DEST_VAR}}")
endfunction()
