#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

set_property(GLOBAL
             PROPERTY LEGATE_BASE_CLANG_TIDY_COMMAND
                      "${LEGATE_CLANG_TIDY}" #
                      --config-file="${LEGATE_DIR}/.clang-tidy" #
                      --use-color #
                      --quiet #
                      --extra-arg=-Wno-error=unused-command-line-argument #
                      --extra-arg=-UNDEBUG #
                      --extra-arg=-DLEGATE_CLANG_TIDY=1 #
                      # Need to define this unconditionally, as otherwise the static
                      # analyzer doesn't see certain assert()'s
                      --extra-arg=-DLEGION_DEBUG=1)

function(_legate_get_tidy_plugin_targets DEST_VAR)
  list(APPEND CMAKE_MESSAGE_CONTEXT "get_tidy_plugin_targets")

  get_property(plugins GLOBAL PROPERTY LEGATE_TIDY_PLUGINS)
  if(NOT plugins AND legate_TIDY_PLUGINS_REQUIRED)
    message(FATAL_ERROR "clang-tidy plugins failed to propagate the "
                        "list of configured plugins.")
  endif()
  set(${DEST_VAR} "${plugins}" PARENT_SCOPE)
endfunction()

function(_legate_create_tidy_plugins)
  # Some package managers (conda) have separate LLVM and clang development packages.
  # Finding just LLVM or just clang does not guarantee that the other is installed, so
  # need to check both exist upfront here.
  find_package(LLVM CONFIG QUIET)
  find_package(Clang CONFIG QUIET HINTS "${LLVM_CMAKE_DIR}/..")
  if((NOT LLVM_FOUND) OR (NOT Clang_FOUND))

    if(NOT LLVM_FOUND)
      set(missing "LLVM")
    else()
      set(missing "Clang")
    endif()

    if(legate_TIDY_PLUGINS_REQUIRED)
      set(severity FATAL_ERROR)
    else()
      set(severity VERBOSE)
    endif()

    message(${severity}
            "Failed to find ${missing}, so cannot build the clang-tidy plugins. "
            "Either point CMake to your ${missing} installation via -D${missing}_DIR=... "
            "and/or install a development build of ${missing}")
    return()
  endif()
  # Use an absolute path for binary directory because we are not certain where this
  # function will be called from the first time. Also, because we need to pass the path to
  # the plugins directly to clang-tidy, we need a stable directory where they will be
  # output.
  set(plugin_dir "${PROJECT_BINARY_DIR}/clang_tidy_plugins")

  add_subdirectory("${LEGATE_DIR}/scripts/pre-commit/clang_tidy_plugins"
                   "${plugin_dir}"
                   # We don't need to build the plugins until the user actually invokes
                   # the tidy targets
                   EXCLUDE_FROM_ALL)

  _legate_get_tidy_plugin_targets(all_plugins)

  foreach(plugin IN LISTS all_plugins)
    set_property(GLOBAL APPEND PROPERTY LEGATE_BASE_CLANG_TIDY_COMMAND
                                        "--load=$<TARGET_FILE:${plugin}>")
  endforeach()
endfunction()

function(_legate_ensure_tidy_target)
  if(TARGET tidy)
    return()
  endif()

  if(LEGATE_CLANG_TIDY)
    add_custom_target(tidy COMMENT "running clang-tidy")

    _legate_create_tidy_plugins()
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

  _legate_ensure_tidy_target()
  if(NOT LEGATE_CLANG_TIDY)
    return()
  endif()

  cmake_path(SET src NORMALIZE "${_TIDY_SOURCE}")
  if(NOT IS_ABSOLUTE "${src}")
    cmake_path(SET src NORMALIZE "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
  endif()

  cmake_path(RELATIVE_PATH src BASE_DIRECTORY "${LEGATE_DIR}" OUTPUT_VARIABLE rel_src)
  string(MAKE_C_IDENTIFIER "${rel_src}_tidy" tidy_target)

  get_property(clang_tidy_cmd GLOBAL PROPERTY LEGATE_BASE_CLANG_TIDY_COMMAND)

  add_custom_target("${tidy_target}"
                    DEPENDS "${src}"
                    COMMAND "${CMAKE_COMMAND}" #
                            -DCLANG_TIDY="${clang_tidy_cmd}" #
                            -DCMAKE_BINARY_DIR="${CMAKE_BINARY_DIR}" #
                            -DSRC="${src}" #
                            -DSED="${LEGATE_SED}" #
                            -P "${LEGATE_CMAKE_DIR}/scripts/clang_tidy.cmake"
                    COMMENT "clang-tidy ${rel_src}"
                    COMMAND_EXPAND_LISTS)

  _legate_get_tidy_plugin_targets(tidy_plugins)
  if(tidy_plugins)
    add_dependencies("${tidy_target}" ${tidy_plugins})
  endif()
  add_dependencies(tidy "${tidy_target}")
endfunction()

function(legate_add_tidy_diff_target)
  list(APPEND CMAKE_MESSAGE_CONTEXT "add_tidy_diff_target")

  set(options)
  set(one_value_args)
  set(multi_value_args)
  cmake_parse_arguments(_TIDY "${options}" "${one_value_args}" "${multi_value_args}"
                        ${ARGN})

  find_package(Git)
  if(LEGATE_CLANG_TIDY_DIFF AND LEGATE_CLANG_TIDY AND LEGATE_SED AND Git_FOUND)
    include(ProcessorCount)

    ProcessorCount(proc_count)

    # cmake-format: off
    add_custom_target(
      tidy-diff
      COMMENT "Running clang-tidy-diff"
      COMMAND
        "${GIT_EXECUTABLE}" diff --no-ext-diff
        -U0 `${GIT_EXECUTABLE} remote show origin \| ${LEGATE_SED} -n "/HEAD branch/s/.*: //p" ` HEAD
        \|
        "${LEGATE_CLANG_TIDY_DIFF}"
        -p 1
        -clang-tidy-binary "${LEGATE_CLANG_TIDY}"
        -path "${CMAKE_BINARY_DIR}"
        -use-color
        -quiet
        -extra-arg=-Wno-error=unused-command-line-argument
        -j "${proc_count}"
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

function(legate_add_external_tidy_target)
  list(APPEND CMAKE_MESSAGE_CONTEXT "add_external_tidy_target")

  set(options)
  set(one_value_args ROOT_DIR)
  set(multi_value_args SOURCES)
  cmake_parse_arguments(_TIDY "${options}" "${one_value_args}" "${multi_value_args}"
                        ${ARGN})

  if(NOT _TIDY_ROOT_DIR)
    message(FATAL_ERROR "Must provide ROOT_DIR option")
  endif()

  if(NOT _TIDY_SOURCES)
    message(FATAL_ERROR "Must provide SOURCES option")
  endif()

  _legate_ensure_tidy_target()
  if(NOT LEGATE_CLANG_TIDY)
    return()
  endif()

  cmake_path(SET root_dir NORMALIZE "${_TIDY_ROOT_DIR}")
  if(NOT IS_ABSOLUTE "${root_dir}")
    cmake_path(SET root_dir NORMALIZE "${CMAKE_CURRENT_SOURCE_DIR}/${root_dir}")
  endif()

  set(absolute_sources)
  foreach(src IN LISTS _TIDY_SOURCES)
    if(NOT IS_ABSOLUTE "${src}")
      cmake_path(SET src NORMALIZE "${root_dir}/${src}")
    endif()
    list(APPEND absolute_sources "${src}")
  endforeach()

  # We make a single target for all sources since we need to configure the project in
  # order to run clang-tidy. If we did a separate target per source, then we'd need to
  # find a way to configure the system only once since cmake doesn't do anything to block
  # concurrent access to a configure tree.
  cmake_path(RELATIVE_PATH root_dir BASE_DIRECTORY "${LEGATE_DIR}" OUTPUT_VARIABLE
             rel_root_dir)
  string(MAKE_C_IDENTIFIER "${rel_root_dir}_tidy" tidy_target)

  cmake_path(SET build_dir NORMALIZE
             "${CMAKE_CURRENT_BINARY_DIR}/external_tidy_targets/${tidy_target}_build")
  file(MAKE_DIRECTORY "${build_dir}")

  get_property(clang_tidy_cmd GLOBAL PROPERTY LEGATE_BASE_CLANG_TIDY_COMMAND)

  add_custom_target("${tidy_target}"
                    DEPENDS "${src}"
                    COMMAND "${CMAKE_COMMAND}" #
                            -DROOT_DIR="${root_dir}" #
                            -DBUILD_DIR="${build_dir}" #
                            -DCLANG_TIDY="${clang_tidy_cmd}" #
                            -DSOURCES="${absolute_sources}" #
                            -DSED="${LEGATE_SED}" #
                            -DLEGATE_BUILD_DIR="${CMAKE_BINARY_DIR}" #
                            -P "${LEGATE_CMAKE_DIR}/scripts/external_clang_tidy.cmake" #
                    COMMENT "clang-tidy ${rel_root_dir}")

  _legate_get_tidy_plugin_targets(tidy_plugins)
  if(tidy_plugins)
    add_dependencies("${tidy_target}" ${tidy_plugins})
  endif()
  add_dependencies(tidy "${tidy_target}")
endfunction()
