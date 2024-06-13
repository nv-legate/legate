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

include(CheckCompilerFlag)
include(CheckLinkerFlag)

function(legate_core_set_default_flags_impl)
  list(APPEND CMAKE_MESSAGE_CONTEXT "set_default_flags")

  set(options SET_CACHE IS_LINKER)
  set(one_value_args DEST_VAR LANG)
  set(multi_value_args FLAGS)

  cmake_parse_arguments(_FLAGS "${options}" "${one_value_args}" "${multi_value_args}"
                        ${ARGN})

  if(NOT _FLAGS_DEST_VAR)
    message(FATAL_ERROR "Must pass DEST_VAR")
  endif()

  if(NOT _FLAGS_LANG)
    message(FATAL_ERROR "Must pass LANG")
  endif()

  if(NOT _FLAGS_FLAGS)
    message(STATUS "No flags to add to ${_FLAGS_DEST_VAR}, bailing")
    return() # nothing to do
  endif()

  get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
  if(NOT ("${_FLAGS_LANG}" IN_LIST languages))
    message(STATUS "Language '${_FLAGS_LANG}' not enabled, bailing")
    return()
  endif()

  set(cmake_req_link_backup "${CMAKE_REQUIRED_LINK_OPTIONS}")
  foreach(flag IN LISTS _FLAGS_FLAGS)
    string(MAKE_C_IDENTIFIER "${flag}" flag_sanitized)

    # The sanitizers need to also have the flag passed to the linker. This is kind of a
    # hack, but oh well.
    if(flag MATCHES "sanitize=.*")
      set(CMAKE_REQUIRED_LINK_OPTIONS "${CMAKE_REQUIRED_LINK_OPTIONS};${flag}")
    endif()

    if(_FLAGS_IS_LINKER)
      check_linker_flag(${_FLAGS_LANG} "${flag}" ${flag_sanitized}_supported)
    else()
      check_compiler_flag(${_FLAGS_LANG} "${flag}" ${flag_sanitized}_supported)
    endif()
    if(${flag_sanitized}_supported)
      message(STATUS "${flag} supported, adding to ${_FLAGS_DEST_VAR}")
      list(APPEND dest "${flag}")
    endif()
    set(CMAKE_REQUIRED_LINK_OPTIONS "${cmake_req_link_backup}")
  endforeach()

  set(${_FLAGS_DEST_VAR} "${dest}" PARENT_SCOPE)
  if(_FLAGS_SET_CACHE)
    set(${_FLAGS_DEST_VAR} "${dest}" CACHE STRING "" FORCE)
  endif()
endfunction()

function(legate_core_configure_default_compiler_flags)
  set(default_cxx_flags_debug
      "-Wall"
      "-Wextra"
      "-Werror"
      "-fstack-protector"
      "-Walloca"
      "-Wdeprecated"
      "-Wimplicit-fallthrough"
      "-fdiagnostics-show-template-tree"
      "-Wignored-qualifiers"
      "-Wmissing-field-initializers"
      "-Wshadow"
      "-pedantic"
      "-Wsign-compare"
      "-Wshadow"
      "-Wshadow-all"
      "-Warray-bounds-pointer-arithmetic"
      "-Wassign-enum"
      "-Wformat-pedantic"
      "-D_LIBCPP_ENABLE_ASSERTIONS=1"
      "-D_LIBCPP_ENABLE_NODISCARD=1")
  set(default_cxx_flags_sanitizer
      "-fsanitize=address,undefined,bounds" "-fno-sanitize-recover=undefined"
      "-fno-omit-frame-pointer" "-g")
  set(default_cxx_flags_release "-O3" "-fstack-protector-strong")
  set(default_cxx_flags_relwithdebinfo ${default_cxx_flags_debug}
                                       ${default_cxx_flags_release})

  macro(set_cxx_flags cxx_flags_var flags_var)
    list(APPEND ${cxx_flags_var} "${${flags_var}}")
    if(legate_core_ENABLE_SANITIZERS)
      list(APPEND ${cxx_flags_var} ${default_cxx_flags_sanitizer})
    endif()
  endmacro()

  set(default_cxx_flags)
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set_cxx_flags(default_cxx_flags default_cxx_flags_debug)
  elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set_cxx_flags(default_cxx_flags default_cxx_flags_release)
  elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set_cxx_flags(default_cxx_flags default_cxx_flags_relwithdebinfo)
  endif()

  if(NOT legate_core_CXX_FLAGS)
    legate_core_set_default_flags_impl(SET_CACHE LANG CXX DEST_VAR legate_core_CXX_FLAGS
                                       FLAGS ${default_cxx_flags})
    set(legate_core_CXX_FLAGS "${legate_core_CXX_FLAGS}" PARENT_SCOPE)
  endif()

  if(NOT legate_core_CUDA_FLAGS)
    set(cuda_flags "${legate_core_CXX_FLAGS}")
    list(REMOVE_ITEM cuda_flags "-pedantic")
    list(REMOVE_ITEM cuda_flags "-Wpedantic")

    foreach(flag IN LISTS cuda_flags)
      list(APPEND default_cuda_flags --compiler-options="${flag}")
    endforeach()

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      list(APPEND default_cuda_flags "-g" "-G")
      # nvcc warning : '--device-debug (-G)' overrides '--generate-line-info (-lineinfo)'
      # ptxas warning : Conflicting options --device-debug and --generate-line-info
      # specified, ignoring --generate-line-info option
      list(REMOVE_ITEM default_cuda_flags "-lineinfo" "--generate-line-info")
    elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
      list(REMOVE_ITEM default_cuda_flags "-G" "--device-debug")
      list(APPEND default_cuda_flags "-g" "-lineinfo")
    endif()

    legate_core_set_default_flags_impl(SET_CACHE LANG CUDA DEST_VAR
                                       legate_core_CUDA_FLAGS FLAGS ${default_cuda_flags})
    set(legate_core_CUDA_FLAGS "${legate_core_CUDA_FLAGS}" PARENT_SCOPE)
  endif()
endfunction()

function(legate_core_configure_default_linker_flags)
  # There are no default linker flags currently.
  set(default_linker_flags)
  if(legate_core_ENABLE_SANITIZERS)
    list(APPEND default_linker_flags "-fsanitize=address,undefined,bounds"
         "-fno-sanitize-recover=undefined")
  endif()

  if(NOT legate_core_LINKER_FLAGS)
    legate_core_set_default_flags_impl(
      SET_CACHE IS_LINKER LANG CXX DEST_VAR legate_core_LINKER_FLAGS FLAGS
      ${default_linker_flags})
    set(legate_core_LINKER_FLAGS "${legate_core_LINKER_FLAGS}" PARENT_SCOPE)
  endif()
endfunction()
