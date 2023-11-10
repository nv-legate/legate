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
  set(options)
  set(oneValueArgs TARGET_NAME TARGET_COMMENT)
  set(multiValueArgs COMMANDS)
  cmake_parse_arguments(_TIDY_TARGET "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  add_custom_target(
    "${_TIDY_TARGET_TARGET_NAME}"
    COMMAND ${_TIDY_TARGET_COMMANDS}
    COMMENT "${_TIDY_TARGET_TARGET_COMMENT}"
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

  macro(search_for_program VARIABLE_NAME PROGRAM_NAME)
    if (${VARIABLE_NAME})
      message("-- Using ${PROGRAM_NAME}: ${${VARIABLE_NAME}}")
    else()
      find_program(${VARIABLE_NAME} ${PROGRAM_NAME})
      if (${VARIABLE_NAME})
        message("-- Found ${PROGRAM_NAME}: ${${VARIABLE_NAME}}")
      endif()
    endif()
  endmacro()

  search_for_program(LEGATE_CORE_RUN_CLANG_TIDY run-clang-tidy)
  search_for_program(LEGATE_CORE_CLANG_TIDY clang-tidy)
  search_for_program(LEGATE_CORE_CLANG_TIDY_DIFF clang-tidy-diff.py)
  search_for_program(LEGATE_CORE_SED sed)
  find_package(Git)

  if (LEGATE_CORE_RUN_CLANG_TIDY AND LEGATE_CORE_CLANG_TIDY)
    _legate_core_add_tidy_target_impl(
      TARGET_NAME tidy
      TARGET_COMMENT "Running clang-tidy"
      COMMANDS
        ${LEGATE_CORE_RUN_CLANG_TIDY}
        -config-file ${CMAKE_CURRENT_SOURCE_DIR}/.clang-tidy
        -clang-tidy-binary ${LEGATE_CORE_CLANG_TIDY}
        -p ${CMAKE_CURRENT_BINARY_DIR}
        -use-color
        -quiet
        ${_TIDY_SOURCES}
    )
  else()
    _legate_core_add_tidy_target_impl(
      TARGET_NAME tidy
      TARGET_COMMENT "Running clang-tidy"
      COMMANDS
              ${CMAKE_COMMAND} -E echo "-- ERROR:"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR: Could not locate 'run-clang-tidy' and/or 'clang-tidy'"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR: Must provide location of both to run clang-tidy target"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR:"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR: Note:"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR: run-clang-tidy is provided by LLVM, so you must have a copy installed locally"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR: Once installed, ensure that path/to/llvm/bin/run-clang-tidy is findable"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR:"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR: Either put path/to/llvm/bin into your PATH (not recommended)"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR: Or symlink path/to/llvm/bin/run-clang-tidy somewhere that is on your PATH (recommended)"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR:"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR: Alternatively, re-run configure with -DLEGATE_CORE_RUN_CLANG_TIDY=/path/to/your/run-clang-tidy"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR: and (if needed) -DLEGATE_CORE_CLANG_TIDY=/path/to/your/clang-tidy"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR:"
      COMMAND ${CMAKE_COMMAND} -E false # to signal the error
    )
  endif()

  if (LEGATE_CORE_CLANG_TIDY_DIFF AND LEGATE_CORE_CLANG_TIDY AND LEGATE_CORE_SED AND Git_FOUND)
    _legate_core_add_tidy_target_impl(
      TARGET_NAME tidy-diff
      TARGET_COMMENT "Running clang-tidy-diff"
      COMMANDS
        ${GIT_EXECUTABLE} diff
        -U0
        `${GIT_EXECUTABLE} remote show origin \| sed -n "/HEAD branch/s/.*: //p"`
        HEAD
        \|
        ${LEGATE_CORE_CLANG_TIDY_DIFF}
        -p 1
        -clang-tidy-binary ${LEGATE_CORE_CLANG_TIDY}
        -path ${CMAKE_CURRENT_BINARY_DIR}
        -use-color
        -quiet
    )
  else()
    _legate_core_add_tidy_target_impl(
      TARGET_NAME tidy-diff
      TARGET_COMMENT "Running clang-tidy-diff"
      COMMANDS
              ${CMAKE_COMMAND} -E echo "-- ERROR:"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR: Could not locate 'clang-tidy-diff.py' and/or 'clang-tidy', 'git', or 'sed'"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR: Must provide location of all to run clang-tidy-diff target"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR:"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR: Note:"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR: clang-tidy-diff.py is provided by LLVM, so you must have a copy installed locally"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR: Once installed, ensure that path/to/llvm/bin/clang-tidy-diff.py is findable"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR:"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR: Either put path/to/llvm/<bin or share/clang> into your PATH (not recommended)"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR: Or symlink path/to/llvm/<bin or share/clang>/clang-tidy-diff.py somewhere that is on your PATH (recommended)"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR:"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR: Alternatively, re-run configure with -DLEGATE_CORE_CLANG_TIDY_DIFF=/path/to/your/clang-tidy-diff.py"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR: and (if needed) -DLEGATE_CORE_CLANG_TIDY=/path/to/your/clang-tidy"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR: and (if needed) -DLEGATE_CORE_SED=/path/to/your/sed"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR:"
      COMMAND ${CMAKE_COMMAND} -E false # to signal the error
    )
  endif()
endfunction()
