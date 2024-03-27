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

cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/policies.cmake NO_POLICY_SCOPE)

list(APPEND CMAKE_MESSAGE_CONTEXT "legate.core")

##############################################################################
# - Setup LEGATE_CORE_DIR and LEGATE_CORE_ARCH -------------------------------

if(NOT DEFINED LEGATE_CORE_DIR)
  # this one we can figure out on our own
  set(LEGATE_CORE_DIR "${CMAKE_CURRENT_LIST_DIR}")
endif()

cmake_path(SET LEGATE_CORE_DIR NORMALIZE "${LEGATE_CORE_DIR}")
# push it to cache
set(
  LEGATE_CORE_DIR "${LEGATE_CORE_DIR}"
  CACHE PATH
  "Full path to the legate.core root directory"
  FORCE
)

if(NOT DEFINED LEGATE_CORE_ARCH)
  set(bin_dir ${CMAKE_CURRENT_BINARY_DIR})
  if("$ENV{AEDIFIX}" STREQUAL "1")
    # We are being driven by Aedifix, that must mean our current binary path is
    # $LEGATE_CORE_DIR/<arch-name>/cmake_build. That being said, this path should never be
    # exercised, because aedifix should be setting this value for us!
    if(NOT "${bin_dir}" MATCHES "cmake_build$")
      message(FATAL_ERROR
        "Could not locate LEGATE_CORE_ARCH directory, expected binary dir to end "
        "with 'cmake_build', but it is ${bin_dir}. Expected this directory structure "
        "because environment variable: AEDIFIX=1. Likely this indicates this environment "
        "variable is wrongly set."
      )
    endif()
    cmake_path(GET bin_dir PARENT_PATH bin_dir)
  endif()
  cmake_path(GET bin_dir STEM LEGATE_CORE_ARCH)
  unset(bin_dir)
endif()

# push it to cache
set(
  LEGATE_CORE_ARCH "${LEGATE_CORE_ARCH}"
  CACHE STRING
  "Name of the legate.core arch directory"
  FORCE
)

cmake_path(SET LEGATE_CORE_ARCH_DIR NORMALIZE "${LEGATE_CORE_DIR}/${LEGATE_CORE_ARCH}")

##############################################################################
# - C++ and CUDA standard ----------------------------------------------------

set(legate_core_CXX_STANDARD 17)
if(CMAKE_CXX_STANDARD)
  if(NOT ${CMAKE_CXX_STANDARD} EQUAL ${legate_core_CXX_STANDARD})
    message(WARNING "CMAKE_CXX_STANDARD set to ${CMAKE_CXX_STANDARD}. Legate.Core expects to be compiled with C++${legate_core_CXX_STANDARD}, ignoring CMAKE_CXX_STANDARD.")
  endif()
endif()
set(CMAKE_CXX_STANDARD ${legate_core_CXX_STANDARD} CACHE STRING "" FORCE)
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE STRING "" FORCE)

if(CMAKE_CUDA_STANDARD)
  if(NOT ${CMAKE_CUDA_STANDARD} EQUAL ${legate_core_CXX_STANDARD})
    message(WARNING "CMAKE_CUDA_STANDARD set to ${CMAKE_CUDA_STANDARD}. Legate.Core expects to be compiled with C++${legate_core_CXX_STANDARD}, ignoring CMAKE_CUDA_STANDARD.")
  endif()
endif()
set(CMAKE_CUDA_STANDARD ${legate_core_CXX_STANDARD} CACHE STRING "" FORCE)
set(CMAKE_CUDA_STANDARD_REQUIRED ON CACHE STRING "" FORCE)
unset(legate_core_CXX_STANDARD) # clean up

##############################################################################
# - Download and initialize RAPIDS CMake helpers -----------------------------

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/include_rapids.cmake)

legate_include_rapids()

macro(set_ifndef variable value)
  if(NOT DEFINED ${variable})
    set(${variable} ${value})
  endif()
endmacro()
# For now we want the optimization flags to match on both normal make and cmake
# builds so we override the cmake defaults here for release, this changes
# -O3 to -O2 and removes -DNDEBUG
set_ifndef(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set_ifndef(CMAKE_CUDA_FLAGS_DEBUG "-O0 -g")
set_ifndef(CMAKE_CXX_FLAGS_RELEASE "-O2")
set_ifndef(CMAKE_CUDA_FLAGS_RELEASE "-O2")
set_ifndef(CMAKE_CXX_FLAGS_MINSIZEREL "-Os")
set_ifndef(CMAKE_CUDA_FLAGS_MINSIZEREL "-Os")
# Order is important, we want the release flags *last*. We want the enable-optimization
# flags to supersede any disable-optimization flags set by the debug flags.
set_ifndef(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS_RELEASE} -g")
set_ifndef(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_DEBUG} ${CMAKE_CUDA_FLAGS_RELEASE} -g")

set(legate_core_version 24.05.00)

if(SKBUILD)
  project(
    legate_core_python
    VERSION ${legate_core_version}
    LANGUAGES # TODO: Building Python extension modules via the python_extension_module requires the C
              # language to be enabled here. The test project that is built in scikit-build to verify
              # various linking options for the python library is hardcoded to build with C, so until
              # that is fixed we need to keep C.
              C CXX)
  include(${CMAKE_CURRENT_SOURCE_DIR}/legate_core_python.cmake)
else()
  project(legate_core VERSION ${legate_core_version} LANGUAGES C CXX)
  include(${CMAKE_CURRENT_SOURCE_DIR}/legate_core_cpp.cmake)
endif()

if(CMAKE_GENERATOR STREQUAL "Ninja")
  function(add_touch_legate_core_ninja_build_target)
    if(SKBUILD)
      set(_suf "_python")
    else()
      set(_suf )
    endif()
    add_custom_target("touch_legate_core${_suf}_ninja_build" ALL
      COMMAND ${CMAKE_COMMAND} -E touch_nocreate "${CMAKE_CURRENT_BINARY_DIR}/build.ninja"
      COMMENT "touch build.ninja so ninja doesn't re-run CMake on rebuild"
      VERBATIM
    )
    foreach(_dep IN ITEMS legion_core legion_core_python
                          Legion LegionRuntime
                          Realm RealmRuntime
                          Regent)
      if(TARGET ${_dep})
        add_dependencies("touch_legate_core${_suf}_ninja_build" ${_dep})
      endif()
    endforeach()
  endfunction()
  add_touch_legate_core_ninja_build_target()
endif()

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
