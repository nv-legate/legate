#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

function(legate_handle_imported_target_rpaths_ ret_val target)
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

function(legate_handle_normal_target_rpaths_ ret_val target)
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

function(legate_populate_dependency_rpaths_editable ret_val)
  list(APPEND CMAKE_MESSAGE_CONTEXT "editable")

  set(legate_cython_rpaths)
  set(legate_targets legate::legate Legion::LegionRuntime Realm::Realm)
  if(TARGET hdf5_vfd_gds)
    list(APPEND legate_targets hdf5_vfd_gds)
  endif()
  # Handle "normal" dependencies which set LIBRARY_OUTPUT_DIRECTORY
  foreach(target IN LISTS legate_targets)
    get_target_property(imported ${target} IMPORTED)
    if(imported)
      legate_handle_imported_target_rpaths_(legate_cython_rpaths ${target})
    else()
      legate_handle_normal_target_rpaths_(legate_cython_rpaths ${target})
    endif()
  endforeach()
  list(REMOVE_DUPLICATES legate_cython_rpaths)
  set(${ret_val} "${legate_cython_rpaths}" PARENT_SCOPE)
endfunction()

function(legate_populate_cython_dependency_rpaths)
  list(APPEND CMAKE_MESSAGE_CONTEXT "populate_cython_dependency_rpaths")

  set(options)
  set(one_value_args RAPIDS_ASSOCIATED_TARGET ROOT_DIRECTORY)
  set(multi_value_args)
  cmake_parse_arguments(
    _LEGATE
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )

  if(NOT _LEGATE_RAPIDS_ASSOCIATED_TARGET)
    message(FATAL_ERROR "Must pass RAPIDS_ASSOCIATED_TARGET")
  endif()

  if(NOT TARGET ${_LEGATE_RAPIDS_ASSOCIATED_TARGET})
    message(
      FATAL_ERROR
      "RAPIDS_ASSOCIATED_TARGET '${_LEGATE_RAPIDS_ASSOCIATED_TARGET}' is not a valid target"
    )
  endif()

  if(SKBUILD_STATE STREQUAL "editable")
    # If we are doing an editable install, then the cython rpaths need to point back to
    # the original (uninstalled) legate libs, since otherwise it cannot find them.
    legate_populate_dependency_rpaths_editable(legate_cython_rpaths)
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
    set(legate_cython_rpaths "../../" "../../legate/deps")
    # The Python PIP wheels layout is quite different to the conda layout.
    if(LEGATE_BUILD_PIP_WHEELS)
      set(
        legate_cython_rpaths
        "${CMAKE_INSTALL_LIBDIR}"
        "${CMAKE_INSTALL_LIBDIR}/legate/deps"
      )
    endif()
  endif()

  message(STATUS "legate_cython_rpaths='${legate_cython_rpaths}'")
  if(_LEGATE_ROOT_DIRECTORY)
    set(root_dir_opts ROOT_DIRECTORY "${_LEGATE_ROOT_DIRECTORY}")
  else()
    set(root_dir_opts)
  endif()

  rapids_cython_add_rpath_entries(
    TARGET "${_LEGATE_RAPIDS_ASSOCIATED_TARGET}"
    PATHS ${legate_cython_rpaths} ${root_dir_opts}
  )

  if(SKBUILD_STATE STREQUAL "editable")
    # In editable mode we do not install the libraries into the wheel, and hence the
    # relative rpaths that are computed by rapids_cython_add_rpath_entries() won't work.
    # What we actually want is to point the rpaths back to the original uninstalled
    # directories
    foreach(path IN LISTS legate_cython_rpaths)
      if(NOT IS_ABSOLUTE "${path}")
        message(FATAL_ERROR "Non-absolute path in editable install: ${path}")
      endif()
    endforeach()

    get_property(cython_targets GLOBAL PROPERTY LEGATE_CYTHON_TARGETS)
    foreach(cy_target IN LISTS cython_targets)
      foreach(path IN LISTS legate_cython_rpaths)
        set_property(TARGET "${cy_target}" APPEND PROPERTY INSTALL_RPATH "${path}")
      endforeach()
    endforeach()
  endif()
endfunction()
