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

##############################################################################
# - Download and initialize RAPIDS CMake helpers -----------------------------

set(legate_core_version 23.11.00)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/legate_helper_functions.cmake)

legate_include_rapids()

# For now we want the optimization flags to match on both normal make and cmake
# builds so we override the cmake defaults here for release, this changes
# -O3 to -O2 and removes -DNDEBUG
if(NOT DEFINED CMAKE_CXX_FLAGS_DEBUG)
  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
endif()
if(NOT DEFINED CMAKE_CUDA_FLAGS_DEBUG)
  set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -g")
endif()
if (NOT CMAKE_CXX_FLAGS_RELEASE)
  # not being set by preset
  set(CMAKE_CXX_FLAGS_RELEASE "-O2")
endif()
if (NOT CMAKE_CUDA_FLAGS_RELEASE)
  # not being set by preset
  set(CMAKE_CUDA_FLAGS_RELEASE "-O2")
endif()
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os")
set(CMAKE_CUDA_FLAGS_MINSIZEREL "-Os")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O2 -g")

if(NOT SKBUILD)
  project(legate_core VERSION ${legate_core_version} LANGUAGES C CXX)
  include(${CMAKE_CURRENT_SOURCE_DIR}/legate_core_cpp.cmake)
else()
  project(
    legate_core_python
    VERSION ${legate_core_version}
    LANGUAGES # TODO: Building Python extension modules via the python_extension_module requires the C
              # language to be enabled here. The test project that is built in scikit-build to verify
              # various linking options for the python library is hardcoded to build with C, so until
              # that is fixed we need to keep C.
              C CXX)
  include(${CMAKE_CURRENT_SOURCE_DIR}/legate_core_python.cmake)
endif()

if(CMAKE_GENERATOR STREQUAL "Ninja")
  function(add_touch_legate_core_ninja_build_target)
    set(_suf )
    if(SKBUILD)
      set(_suf "_python")
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
