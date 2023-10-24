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

function(_legate_core_add_tidy_target_impl)
  add_custom_target(
    tidy
    COMMAND ${ARGV}
    COMMENT "Running clang-tidy"
    COMMAND_EXPAND_LISTS
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
endfunction()

function(legate_core_add_tidy_target)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs SOURCES)
  cmake_parse_arguments(_TIDY "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT _TIDY_SOURCES)
    message(FATAL_ERROR "Must provide SOURCES option")
  endif()

  find_program(LEGATE_CORE_RUN_CLANG_TIDY run-clang-tidy)
  if (LEGATE_CORE_RUN_CLANG_TIDY)
    _legate_core_add_tidy_target_impl(
      ${LEGATE_CORE_RUN_CLANG_TIDY}
      -config-file=${CMAKE_CURRENT_SOURCE_DIR}/.clang-tidy
      -p ${CMAKE_CURRENT_BINARY_DIR}
      -use-color
      -quiet
      -header-filter "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
      ${_TIDY_SOURCES}
    )
  else()
    _legate_core_add_tidy_target_impl(
              ${CMAKE_COMMAND} -E echo "-- ERROR:"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR: Could not locate 'run-clang-tidy'"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR: Must provide location of it to run clang-tidy target"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR:"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR: Note:"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR: run-clang-tidy is provided by LLVM, so you must have a copy installed locally"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR: Once installed, ensure that path/to/llvm/bin/run-clang-tidy is findable"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR:"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR: Either put path/to/llvm/bin into your PATH (not recommended)"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR: Or symlink path/to/llvm/bin/run-clang-tidy somewhere that is on your PATH (recommended)"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR:"
      COMMAND ${CMAKE_COMMAND} -E false # to signal the error
    )
  endif()
endfunction()
