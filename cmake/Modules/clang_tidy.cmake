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

include(${CMAKE_CURRENT_LIST_DIR}/utilities.cmake)

legate_find_program(LEGATE_CLANG_TIDY clang-tidy)
legate_find_program(LEGATE_SED sed)
legate_find_program(LEGATE_CLANG_TIDY_DIFF clang-tidy-diff.py)
if(NOT LEGATE_CLANG_TIDY_DIFF)
  # Sometimes this is not installed under the usual [s]bin directories, but instead under
  # share/clang, so try that as well
  legate_find_program(LEGATE_CLANG_TIDY_DIFF clang-tidy-diff.py
                      FIND_PROGRAM_ARGS PATH_SUFFIXES "share/clang")
endif()

function(legate_add_tidy_target)
  list(APPEND CMAKE_MESSAGE_CONTEXT "add_tidy_target")

  set(options)
  set(one_value_args SOURCE)
  set(multi_value_args)
  cmake_parse_arguments(_TIDY "${options}" "${one_value_args}" "${multi_value_args}"
                        ${ARGN})

  if(NOT _TIDY_SOURCE)
    message(FATAL_ERROR "Must provide SOURCE option")
  endif()

  if(LEGATE_CLANG_TIDY)
    if(NOT TARGET tidy)
      add_custom_target(tidy COMMENT "running clang-tidy")
    endif()

    cmake_path(SET src NORMALIZE "${_TIDY_SOURCE}")
    string(MAKE_C_IDENTIFIER "${src}_tidy" tidy_target)

    if(NOT IS_ABSOLUTE "${src}")
      cmake_path(SET src NORMALIZE "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
    endif()
    cmake_path(RELATIVE_PATH src BASE_DIRECTORY "${LEGATE_DIR}" OUTPUT_VARIABLE rel_src)

    add_custom_target("${tidy_target}"
                      COMMAND "${LEGATE_CLANG_TIDY}"
                              --config-file="${LEGATE_DIR}/.clang-tidy" -p
                              "${CMAKE_BINARY_DIR}" --use-color --quiet
                              --extra-arg=-Wno-error=unused-command-line-argument "${src}"
                      COMMENT "clang-tidy ${rel_src}")

    add_dependencies(tidy "${tidy_target}")
    return()
  endif()

  if(NOT TARGET tidy)
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

function(legate_add_tidy_diff_target)
  list(APPEND CMAKE_MESSAGE_CONTEXT "add_tidy_diff_target")

  set(options)
  set(one_value_args)
  set(multi_value_args)
  cmake_parse_arguments(_TIDY "${options}" "${one_value_args}" "${multi_value_args}"
                        ${ARGN})

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
      WORKING_DIRECTORY "${LEGATE_DIR}"
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
