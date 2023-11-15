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

find_package(Python3 REQUIRED COMPONENTS Development)

function(legate_core_create_cython_modules)
  set(options)
  set(one_value_args)
  set(multi_value_keywords SOURCES)
  cmake_parse_arguments(_LEGATE_CORE_CYTHON "${options}" "${one_value_args}" "${multi_value_keywords}" ${ARGN})

  if (NOT _LEGATE_CORE_CYTHON_SOURCES)
    message(FATAL_ERROR "Must pass SOURCES source1.pyx [source2.pyx ... sourceN.pyx]")
  endif()

  rapids_cython_create_modules(
    CXX
    ASSOCIATED_TARGETS legate_core
    SOURCE_FILES "${_LEGATE_CORE_CYTHON_SOURCES}"
    LINKED_LIBRARIES legate::core
  )

  foreach(target IN LISTS RAPIDS_CYTHON_CREATED_TARGETS)
    target_compile_features(${target} PRIVATE cxx_std_17)
    # Mark python headers as system (in case user has added warning flags, we don't want the
    # python headers to trigger them)
    target_include_directories(${target} SYSTEM PRIVATE ${Python3_INCLUDE_DIRS})
  endforeach()
endfunction()

add_subdirectory(data)
add_subdirectory(mapping)
add_subdirectory(operation)
add_subdirectory(partitioning)
add_subdirectory(runtime)
add_subdirectory(type)
add_subdirectory(utilities)
add_subdirectory(task)

set(
  cython_sources
  legate_c.pyx
)

legate_core_create_cython_modules(SOURCES "${cython_sources}")
