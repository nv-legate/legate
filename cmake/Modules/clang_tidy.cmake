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

function(legate_search_for_program VARIABLE_NAME PROGRAM_NAME)
  message(CHECK_START "Searching for ${PROGRAM_NAME}")

  if(${VARIABLE_NAME})
    message(CHECK_PASS "using pre-found: ${${VARIABLE_NAME}}")
    return()
  endif()

  find_program(${VARIABLE_NAME} ${PROGRAM_NAME} ${ARGN})
  if(${VARIABLE_NAME})
    message(CHECK_PASS "found: ${${VARIABLE_NAME}}")
  else()
    message(CHECK_FAIL "not found")
  endif()
endfunction()

function(_legate_add_tidy_target_impl CLANG_TIDY SOURCES_VAR)
  if(CLANG_TIDY)
    list(REMOVE_DUPLICATES ${SOURCES_VAR})
    foreach(src IN LISTS ${SOURCES_VAR})
      string(MAKE_C_IDENTIFIER "${src}_tidy" src_tidy)
      add_custom_target("${src_tidy}"
                        COMMAND "${CLANG_TIDY}"
                                --config-file="${CMAKE_CURRENT_SOURCE_DIR}/.clang-tidy" -p
                                "${CMAKE_BINARY_DIR}" --use-color --quiet
                                --extra-arg=-Wno-error=unused-command-line-argument
                                "${src}"
                        COMMENT "clang-tidy ${src}"
                        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
      list(APPEND tidy_targets "${src_tidy}")
    endforeach()
    add_custom_target(tidy COMMENT "running clang-tidy")
    if(tidy_targets) # in case it's empty
      add_dependencies(tidy ${tidy_targets})
    endif()
  else()
    # cmake-format: off
    add_custom_target(
      tidy
      COMMENT "Running clang-tidy"
      VERBATIM
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR:"
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
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR: Alternatively, re-run configure with -DLEGATE_RUN_CLANG_TIDY=/path/to/your/run-clang-tidy"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR: and (if needed) -DLEGATE_CLANG_TIDY=/path/to/your/clang-tidy"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR:"
      COMMAND ${CMAKE_COMMAND} -E false # to signal the error
    )
    # cmake-format: on
  endif()
endfunction()

function(_legate_add_tidy_diff_target_impl CLANG_TIDY)
  legate_search_for_program(LEGATE_SED sed)
  legate_search_for_program(LEGATE_CLANG_TIDY_DIFF clang-tidy-diff.py)
  if(NOT LEGATE_CLANG_TIDY_DIFF)
    # Sometimes this is not installed under the usual [s]bin directories, but instead
    # under share/clang, so try that as well
    legate_search_for_program(LEGATE_CLANG_TIDY_DIFF clang-tidy-diff.py PATH_SUFFIXES
                              "share/clang")
  endif()
  find_package(Git)

  if(LEGATE_CLANG_TIDY_DIFF AND CLANG_TIDY AND LEGATE_SED AND Git_FOUND)
    include(ProcessorCount)

    ProcessorCount(PROC_COUNT)
    set(TIDY_PARALLEL_FLAGS "-j${PROC_COUNT}")

    # cmake-format: off
    add_custom_target(
      tidy-diff
      COMMENT "Running clang-tidy-diff"
      COMMAND
        "${GIT_EXECUTABLE}" diff
        -U0 `${GIT_EXECUTABLE} remote show origin \| ${LEGATE_SED} -n "/HEAD branch/s/.*: //p" ` HEAD
        \|
        "${LEGATE_CLANG_TIDY_DIFF}"
        -p 1
        -clang-tidy-binary "${CLANG_TIDY}"
        -path "${CMAKE_BINARY_DIR}"
        -use-color
        -quiet
        -extra-arg=-Wno-error=unused-command-line-argument
        ${TIDY_PARALLEL_FLAGS}
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    # cmake-format: on
  else()
    # cmake-format: off
    add_custom_target(
      tidy-diff
      COMMENT "Running clang-tidy-diff"
      VERBATIM
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR:"
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
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR: Alternatively, re-run configure with -DLEGATE_CLANG_TIDY_DIFF=/path/to/your/clang-tidy-diff.py"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR: and (if needed) -DLEGATE_CLANG_TIDY=/path/to/your/clang-tidy"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR: and (if needed) -DLEGATE_SED=/path/to/your/sed"
      COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR:"
      COMMAND ${CMAKE_COMMAND} -E false # to signal the error
    )
    # cmake-format: on
  endif()
endfunction()

function(legate_add_tidy_target)
  list(APPEND CMAKE_MESSAGE_CONTEXT "add_tidy_target")

  set(options)
  set(one_value_args)
  set(multi_value_args SOURCES)
  cmake_parse_arguments(_TIDY "${options}" "${one_value_args}" "${multi_value_args}"
                        ${ARGN})

  if(NOT _TIDY_SOURCES)
    message(FATAL_ERROR "Must provide SOURCES option")
  endif()

  legate_search_for_program(LEGATE_CLANG_TIDY clang-tidy)

  _legate_add_tidy_target_impl(${LEGATE_CLANG_TIDY} _TIDY_SOURCES)
  _legate_add_tidy_diff_target_impl(${LEGATE_CLANG_TIDY})
endfunction()
