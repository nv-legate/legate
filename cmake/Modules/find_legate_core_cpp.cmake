#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

macro(find_legate_core_cpp_impl legate_core_version build_export_set install_export_set)
  include("${rapids-cmake-dir}/export/detail/parse_version.cmake")
  rapids_export_parse_version(${legate_core_version} legate_core parsed_ver)

  macro(legate_rapids_find_legate)
    message(STATUS "Searching for Legate.Core")
    rapids_find_package(legate_core
      GLOBAL_TARGETS     legate::core
      BUILD_EXPORT_SET   ${build_export_set}
      INSTALL_EXPORT_SET ${install_export_set}
      FIND_ARGS
        ${parsed_ver} EXACT CONFIG REQUIRED
    )
    message(STATUS "legate_core_FOUND = ${legate_core_FOUND}")
    message(STATUS "legate_core_ROOT  = ${legate_core_ROOT}")
  endmacro()

  # We are required to find legate.core if either the user sets legate_core_ROOT, or we
  # have done this before and found legate.core via some pre-installed version.
  if(
      (DEFINED legate_core_ROOT)
      OR (
        (DEFINED _legate_core_FOUND_METHOD) # this means we've been here before
        AND (_legate_core_FOUND_METHOD STREQUAL  "INSTALLED")
      )
    )
    legate_rapids_find_legate()
    set(_legate_core_FOUND_METHOD "INSTALLED")
  endif()

  if((NOT legate_core_FOUND) OR (_legate_core_FOUND_METHOD STREQUAL "PRE_BUILT"))
    cmake_path(SET legate_core_ROOT NORMALIZE "${LEGATE_CORE_ARCH_DIR}/cmake_build")
    # HACK: We need to be able to detect whether the legate core directory contains a
    # **built** version of the library or just a **configured** one. We need to do this
    # because if we just do find_package(legate_core), it will import all of the (possibly
    # not built) targets and we are later met with errors:
    #
    # ninja: error: '/path/to/cmake_build/lib/liblgcore.dylib', needed by
    # 'legate/core/_lib/legate_c.cpython-311-darwin.so', missing and no known rule to make
    # it
    #
    # Unfortunately, cmake generates all the "install" files
    # (e.g. <PackageName>Config.cmake) at configure-time, and does not generate any special
    # markers during build time. So we need to manually search for liblgcore using
    # find_library(). If we find it, we can be reasonably sure that the build directory is
    # usable.
    message(STATUS "Searching ${legate_core_ROOT} for pre-built Legate.core")
    find_library(legate_core_cpp_lib
      NAMES
        core
        lgcore
        liblgcore
        "liblgcore${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "liblgcore${legate_core_version}${CMAKE_SHARED_LIBRARY_SUFFIX}"
      PATHS
        "${legate_core_ROOT}"
        "${legate_core_ROOT}/${lib_dir}"
        "${legate_core_ROOT}/${CMAKE_INSTALL_LIBDIR}"
      NO_DEFAULT_PATH
    )

    if(EXISTS "${legate_core_cpp_lib}")
      message(STATUS "Legate.core appears to already have been built")
      # Found via pre-built, let's ensure those libs are up-to-date before we try to find it
      execute_process(
        COMMAND ${CMAKE_COMMAND} --build .
        WORKING_DIRECTORY "${legate_core_ROOT}"
        COMMAND_ERROR_IS_FATAL ANY
      )
      legate_rapids_find_legate()
      set(_legate_core_FOUND_METHOD "PRE_BUILT")
    elseif(_legate_core_FOUND_METHOD STREQUAL "PRE_BUILT")
      message(FATAL_ERROR "Failed to find legate.core C++ build even though we apparently used-it previously")
    endif()
    unset(legate_core_ROOT) # undo this
  endif()

  if(NOT legate_core_FOUND)
    set(SKBUILD OFF)
    set(Legion_USE_Python ON)
    # These are the names of the variables that we want the C++ build to export up to
    # us.
    #
    # Normally this is done transparently if we find the package above (via the final code
    # snippet embedded in the Findlegate_core.cmake), but if we build it ourselves, then we
    # need legate_core_cpp.cmake to explicitly set(<the variable> ... PARENT_SCOPE) in order
    # for us to see it...
    set(legate_core_SUBDIR_CMAKE_EXPORT_VARS
      "Legion_USE_CUDA"
      "Legion_USE_Python"
      "Legion_USE_OpenMP"
      "Legion_BOUNDS_CHECKS"
      "Legion_MAX_DIM"
      "Legion_MAX_FIELDS"
      "GASNet_CONDUIT"
      CACHE INTERNAL "" FORCE
    )
    add_subdirectory(. legate_core_cpp)
    set(SKBUILD ON)
    set(_legate_core_FOUND_METHOD "SELF_BUILT")
  endif()
endmacro()

macro(find_legate_core_cpp)
  list(APPEND CMAKE_MESSAGE_CONTEXT "find_legate_core_cpp")

  set(one_value_args VERSION BUILD_EXPORT_SET INSTALL_EXPORT_SET)
  cmake_parse_arguments(_FIND_LEGATE_CORE "" "${one_value_args}" "" ${ARGN})

  find_legate_core_cpp_impl(
    ${_FIND_LEGATE_CORE_VERSION}
    ${_FIND_LEGATE_CORE_BUILD_EXPORT_SET}
    ${_FIND_LEGATE_CORE_INSTALL_EXPORT_SET}
  )

  set(_legate_core_FOUND_METHOD ${_legate_core_FOUND_METHOD} CACHE INTERNAL "" FORCE)
  message(STATUS "legate_core_FOUND_METHOD: '${_legate_core_FOUND_METHOD}'")

  # cleanup
  unset(one_value_args)
  unset(_FIND_LEGATE_CORE_VERSION)
  unset(_FIND_LEGATE_CORE_BUILD_EXPORT_SET)
  unset(_FIND_LEGATE_CORE_INSTALL_EXPORT_SET)

  list(POP_BACK CMAKE_MESSAGE_CONTEXT)
endmacro()
