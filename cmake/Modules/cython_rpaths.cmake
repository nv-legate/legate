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

function(legate_core_handle_imported_target_rpaths_ ret_val target)
  if(NOT TARGET ${target})
    message(FATAL_ERROR "${target} is not a target")
  endif()

  # If the target is imported, then LOCATION is set on it:
  #
  # extra_cython_lib_dir = /path/to/<target>/build/dir/lib/lib<target>.so
  get_target_property(extra_cython_rpath ${target} LOCATION)

  if(NOT extra_cython_rpath)
    message(FATAL_ERROR "Could not determine RPATH location of imported target ${target}")
  endif()

  # /path/to/<target>/build/dir/lib/
  cmake_path(GET extra_cython_rpath PARENT_PATH extra_cython_rpath)
  message(VERBOSE "adding rpath for imported target ${target}: ${extra_cython_rpath}")
  # These might be duplicated, we will deal with that later
  list(APPEND ${ret_val} "${extra_cython_rpath}")
  set(${ret_val} "${${ret_val}}" PARENT_SCOPE)
endfunction()

function(legate_core_handle_normal_target_rpaths_ ret_val target)
  # If the target is "regular", i.e. we will build it ourselves, then it suffices to check
  # the library output directory for it, since that will be the location of the libs after
  # we have built them
  if(NOT TARGET ${target})
    message(FATAL_ERROR "${target} is not a target")
  endif()

  # Could be 'lib' or could be /full/path/to/<target>/lib
  get_target_property(extra_cython_rpath ${target} LIBRARY_OUTPUT_DIRECTORY)
  if(NOT extra_cython_rpath)
    message(FATAL_ERROR "Could not determine library output directory for ${target}")
  endif()
  if(NOT IS_ABSOLUTE ${extra_cython_rpath})
    get_target_property(target_bin_dir ${target} BINARY_DIR)

    if(NOT target_bin_dir)
      message(FATAL_ERROR "Could not determine binary dir for ${target}")
    endif()

    cmake_path(SET extra_cython_rpath NORMALIZE "${target_bin_dir}/${extra_cython_rpath}")
  endif()

  message(VERBOSE "adding rpath for normal target ${target}: ${extra_cython_rpath}")
  # These might be duplicated, we will deal with that later
  list(APPEND ${ret_val} "${extra_cython_rpath}")
  set(${ret_val} "${${ret_val}}" PARENT_SCOPE)
endfunction()

function(legate_core_populate_dependency_rpaths_editable ret_val)
  list(APPEND CMAKE_MESSAGE_CONTEXT "editable")

  set(legate_cython_rpaths)
  # Handle "normal" dependencies which set LIBRARY_OUTPUT_DIRECTORY
  foreach(target legate::core Legion::LegionRuntime Legion::RealmRuntime)
    get_target_property(imported ${target} IMPORTED)
    if(imported)
      legate_core_handle_imported_target_rpaths_(legate_cython_rpaths ${target})
    else()
      legate_core_handle_normal_target_rpaths_(legate_cython_rpaths ${target})
    endif()
  endforeach()

  # Handle fmt, which does not set LIBRARY_OUTPUT_DIRECTORY, but we know that it will
  # place its libraries at BINARY_DIR
  get_target_property(imported fmt::fmt IMPORTED)
  if(imported)
    legate_core_handle_imported_target_rpaths_(legate_cython_rpaths fmt::fmt)
  else()
    get_target_property(fmt_bin_dir fmt::fmt BINARY_DIR)
    if(NOT fmt_bin_dir)
      message(FATAL_ERROR "Could not determine binary dir for fmt")
    endif()

    if(NOT IS_ABSOLUTE ${fmt_bin_dir})
      message(FATAL_ERROR "fmt binary dir is not absolute: ${fmt_bin_dir}")
    endif()

    message(VERBOSE "adding rpath for normal target fmt::fmt: ${fmt_bin_dir}")
    list(APPEND legate_cython_rpaths "${fmt_bin_dir}")
  endif()

  list(REMOVE_DUPLICATES legate_cython_rpaths)
  set(${ret_val} "${legate_cython_rpaths}" PARENT_SCOPE)
endfunction()

function(legate_core_populate_cython_dependency_rpaths)
  list(APPEND CMAKE_MESSAGE_CONTEXT "populate_cython_dependency_rpaths")

  set(options)
  set(one_value_args RESULT_VAR)
  set(multi_value_args)
  cmake_parse_arguments(_LEGATE_CORE "${options}" "${one_value_args}"
                        "${multi_value_args}" ${ARGN})

  if(legate_core_SETUP_PY_MODE STREQUAL "develop")
    # If we are doing an editable install, then the cython rpaths need to point back to
    # the original (uninstalled) legate libs, since otherwise it cannot find them.
    legate_core_populate_dependency_rpaths_editable(legate_cython_rpaths)
  else()
    # This somehow sets the rpath correctly for regular installs.
    # rapids_cython_add_rpath_entries() mentions that:
    #
    # PATHS may either be absolute or relative to the ROOT_DIRECTORY. The paths are always
    # converted to be relative to the current directory i.e relative to $ORIGIN in the
    # RPATH.
    #
    # where
    #
    # ROOT_DIRECTORY "Defaults to ${PROJECT_SOURCE_DIR}".
    #
    # Since there is nothing interesting 2 directories up from PROJECT_SOURCE_DIR, my best
    # guess is that the 2 directories up refers to 2 directories up from the python
    # site-packages dir, which is always found as
    # /path/to/lib/python3.VERSION/site-packages/. The combined rpaths would make this
    # point to /path/to/lib which seems right. But who knows.
    set(legate_cython_rpaths "../../")
  endif()

  message(STATUS "legate_cython_rpaths='${legate_cython_rpaths}'")
  set(${_LEGATE_CORE_RESULT_VAR} "${legate_cython_rpaths}" PARENT_SCOPE)
endfunction()
