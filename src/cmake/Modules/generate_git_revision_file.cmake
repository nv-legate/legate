#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

function(legate_generate_git_revision_file)
  list(APPEND CMAKE_MESSAGE_CONTEXT "generate_git_revision_file")

  set(options)
  set(one_value_args GENERATED_SRC_VAR)
  set(multi_value_args EXTRA_TARGETS EXTRA_TARGETS_PREFIX)
  cmake_parse_arguments(_LEGATE "${options}" "${one_value_args}" "${multi_value_args}"
                        ${ARGN})

  foreach(var IN LISTS one_value_args multi_value_args)
    if(NOT _LEGATE_${var})
      message(FATAL_ERROR "Must pass ${var}")
    endif()
  endforeach()

  list(LENGTH _LEGATE_EXTRA_TARGETS num_targets)
  list(LENGTH _LEGATE_EXTRA_TARGETS_PREFIX num_prefix)
  if(NOT num_targets EQUAL num_prefix)
    message(FATAL_ERROR "Must pass same number of prefixes as targets: "
                        "(${num_targets} targets, ${num_prefix} prefixes)")
  endif()

  find_package(Git QUIET REQUIRED)

  set(all_dest_files)
  set(all_cache_files)

  list(PREPEND _LEGATE_EXTRA_TARGETS_PREFIX "LEGATE")
  foreach(prefix IN LISTS _LEGATE_EXTRA_TARGETS_PREFIX)
    string(TOLOWER "${prefix}" prefix_lower)

    set(dest_file "${CMAKE_CURRENT_BINARY_DIR}/generated/${prefix_lower}_git_revision.cc")
    set(cache_file "${dest_file}.git_hash_cache")

    list(APPEND all_dest_files "${dest_file}")
    list(APPEND all_cache_files "${cache_file}")
  endforeach()

  set(all_src_dirs "${CMAKE_CURRENT_SOURCE_DIR}")
  foreach(target IN LISTS _LEGATE_EXTRA_TARGETS)
    get_target_property(target_src_dir ${target} SOURCE_DIR)
    list(APPEND all_src_dirs "${target_src_dir}")
  endforeach()

  set(template_file "${LEGATE_CMAKE_DIR}/templates/git_version.cc.in")
  add_custom_target(generate_git_revisions ALL
                    DEPENDS "${template_file}"
                    BYPRODUCTS ${all_dest_files} ${all_cache_files}
                    COMMAND ${CMAKE_COMMAND} -DGIT_EXECUTABLE="${GIT_EXECUTABLE}"
                            -DTEMPLATE_FILE="${template_file}"
                            -DSRC_DIRS="${all_src_dirs}"
                            -DPREFIXES="${_LEGATE_EXTRA_TARGETS_PREFIX}"
                            -DDEST_FILES="${all_dest_files}"
                            -DCACHE_FILES="${all_cache_files}" -P
                            "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../scripts/git_revision.cmake"
                    COMMENT "Checking the git repository for changes...")

  list(APPEND "${_LEGATE_GENERATED_SRC_VAR}" ${all_dest_files})
  set(${_LEGATE_GENERATED_SRC_VAR} "${${_LEGATE_GENERATED_SRC_VAR}}" PARENT_SCOPE)
endfunction()
