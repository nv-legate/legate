#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

macro(find_legate_cpp_impl legate_version build_export_set install_export_set)
  include("${rapids-cmake-dir}/export/detail/parse_version.cmake")
  rapids_export_parse_version(${legate_version} legate parsed_ver)

  macro(legate_rapids_find_legate)
    message(STATUS "Searching for Legate")
    rapids_find_package(
      legate
      ${parse_ver}
      EXACT
      CONFIG
      REQUIRED
      GLOBAL_TARGETS legate::legate
      BUILD_EXPORT_SET ${build_export_set}
      INSTALL_EXPORT_SET ${install_export_set}
    )
    message(STATUS "legate_FOUND = ${legate_FOUND}")
    message(STATUS "legate_ROOT  = ${legate_ROOT}")
  endmacro()

  # We are required to find legate if either the user sets legate_ROOT, or we have done
  # this before and found legate via some pre-installed version.
  if(
    (DEFINED legate_ROOT)
    OR
      (
        (DEFINED _legate_FOUND_METHOD) # this means we've been here before
        AND (_legate_FOUND_METHOD STREQUAL "INSTALLED")
      )
  )
    legate_rapids_find_legate()
    set(_legate_FOUND_METHOD "INSTALLED")
  endif()

  if((NOT legate_FOUND) OR (_legate_FOUND_METHOD STREQUAL "PRE_BUILT"))
    cmake_path(SET legate_ROOT NORMALIZE "${LEGATE_ARCH_DIR}/cmake_build")
    # HACK: We need to be able to detect whether the legate directory contains a **built**
    # version of the library or just a **configured** one. We need to do this because if
    # we just do find_package(legate), it will import all of the (possibly not built)
    # targets and we are later met with errors:
    #
    # ninja: error: '/path/to/cmake_build/lib/liblegate.dylib', needed by
    # 'legate/_lib/legate_c.cpython-311-darwin.so', missing and no known rule to make it
    #
    # Unfortunately, cmake generates all the "install" files (e.g.
    # <PackageName>Config.cmake) at configure-time, and does not generate any special
    # markers during build time. So we need to manually search for liblegate using
    # find_library(). If we find it, we can be reasonably sure that the build directory is
    # usable.
    message(STATUS "Searching ${legate_ROOT} for pre-built Legate")
    find_library(
      legate_cpp_lib
      NAMES legate
      PATHS "${legate_ROOT}/cpp" "${legate_ROOT}"
      PATH_SUFFIXES lib "${CMAKE_INSTALL_LIBDIR}"
      NO_DEFAULT_PATH
    )

    if(EXISTS "${legate_cpp_lib}")
      message(STATUS "Legate appears to already have been built")
      # Found via pre-built, let's ensure those libs are up-to-date before we try to find
      # it
      include(ProcessorCount)

      ProcessorCount(procs)
      if("${procs}" STREQUAL "0") # some kind of problem occurred
        set(procs 1)
      endif()
      execute_process(
        COMMAND ${CMAKE_COMMAND} --build . -j "${procs}"
        WORKING_DIRECTORY "${legate_ROOT}"
        COMMAND_ERROR_IS_FATAL ANY
      )
      legate_rapids_find_legate()
      set(_legate_FOUND_METHOD "PRE_BUILT")
    elseif(_legate_FOUND_METHOD STREQUAL "PRE_BUILT")
      message(
        FATAL_ERROR
        "Failed to find legate C++ build even though we apparently used-it previously"
      )
    endif()
    unset(legate_ROOT) # undo this
  endif()

  if(NOT legate_FOUND)
    message(STATUS "Legate not found, building from source")
    set(_legate_skbuild_bkp ${SKBUILD})
    set(SKBUILD OFF)
    set(Legion_USE_Python ON)
    set(msg_ctx_back "${CMAKE_MESSAGE_CONTEXT}")
    set(CMAKE_MESSAGE_CONTEXT "--")
    add_subdirectory("${LEGATE_DIR}/src" legate_cpp)
    set(CMAKE_MESSAGE_CONTEXT "${msg_ctx_back}")
    set(SKBUILD ${_legate_skbuild_bkp})
    set(_legate_FOUND_METHOD "SELF_BUILT")
  endif()
endmacro()

macro(find_legate_cpp)
  list(APPEND CMAKE_MESSAGE_CONTEXT "find_legate_cpp")

  set(one_value_args VERSION BUILD_EXPORT_SET INSTALL_EXPORT_SET)
  cmake_parse_arguments(_FIND_LEGATE "" "${one_value_args}" "" ${ARGN})

  find_legate_cpp_impl(
    ${_FIND_LEGATE_VERSION}
    ${_FIND_LEGATE_BUILD_EXPORT_SET}
    ${_FIND_LEGATE_INSTALL_EXPORT_SET}
  )

  set(_legate_FOUND_METHOD ${_legate_FOUND_METHOD} CACHE INTERNAL "" FORCE)
  message(STATUS "legate_FOUND_METHOD: '${_legate_FOUND_METHOD}'")

  # cleanup
  unset(one_value_args)
  unset(_FIND_LEGATE_VERSION)
  unset(_FIND_LEGATE_BUILD_EXPORT_SET)
  unset(_FIND_LEGATE_INSTALL_EXPORT_SET)

  list(POP_BACK CMAKE_MESSAGE_CONTEXT)
endmacro()
