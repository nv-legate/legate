#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

function(legate_install_dependencies)
  list(APPEND CMAKE_MESSAGE_CONTEXT "install_dependency")

  set(one_value_args "FIXUP_TARGET")
  set(multi_value_args "TARGETS")
  cmake_parse_arguments(_LEGATE "" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT _LEGATE_TARGETS)
    message(FATAL_ERROR "Must pass TARGETS")
  endif()

  foreach(target IN LISTS _LEGATE_TARGETS)
    if(NOT TARGET ${target})
      message(FATAL_ERROR "Target ${target} is not a target")
    endif()

    get_target_property(imported ${target} IMPORTED)
    if(imported)
      continue()
    endif()

    get_target_property(base_target ${target} ALIASED_TARGET)
    if(base_target)
      set(target ${base_target})
    endif()

    if(_LEGATE_FIXUP_TARGET)
      cmake_language(CALL "${_LEGATE_FIXUP_TARGET}" ${target})
    endif()

    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.27.0")
      set(archive_namelink_skip NAMELINK_SKIP)
    else()
      set(archive_namelink_skip)
    endif()

    # Note, we want to skip namelink installations because:
    #
    # 1. We are the only consumers of these .so's and CMake will make sure we link directly
    #    to libfoo.so.1.2.3, not the generic libfoo.so.
    # 2. If we are installing into pip wheels, then pip will actually make a deep copy of
    #    every dependency that has a namelink component because wheels don't support
    #    symlink. You effectively give it a list of names, and it deep-copies every name
    #    from src to dest.
    install(
      TARGETS "${target}"
      ARCHIVE DESTINATION "${legate_DEP_INSTALL_LIBDIR}"
      ${archive_namelink_skip}
      LIBRARY DESTINATION "${legate_DEP_INSTALL_LIBDIR}" NAMELINK_SKIP
      RUNTIME DESTINATION "${legate_DEP_INSTALL_BINDIR}"
      PUBLIC_HEADER DESTINATION "${legate_DEP_INSTALL_INCLUDEDIR}"
      PRIVATE_HEADER DESTINATION "${legate_DEP_INSTALL_INCLUDEDIR}"
      INCLUDES DESTINATION "${legate_DEP_INSTALL_INCLUDEDIR}"
    )

    get_target_property(target_type "${target}" TYPE)
    if(target_type STREQUAL "SHARED_LIBRARY")
      set(install_dir "${legate_DEP_INSTALL_LIBDIR}")
    elseif(target_type STREQUAL "EXECUTABLE")
      set(install_dir "${legate_DEP_INSTALL_BINDIR}")
    else()
      set(install_dir "")
    endif()

    if(install_dir)
      legate_install_debug_symbols(TARGET "${target}" INSTALL_DIR "${install_dir}")
    endif()
  endforeach()
endfunction()

macro(legate_export_variables PACKAGE)
  cpm_export_variables(${PACKAGE})
  set(${PACKAGE}_VERSION "${${PACKAGE}_VERSION}" PARENT_SCOPE)
  foreach(
    suffix
    MAJOR
    MINOR
    PATCH
    TWEAK
    COUNT
  )
    if(DEFINED ${PACKAGE}_VERSION_${suffix})
      set(${PACKAGE}_VERSION_${suffix} "${${PACKAGE}_VERSION_${suffix}}" PARENT_SCOPE)
    endif()
  endforeach()
endmacro()

# This guy needs to be a macro in case the find_or_configure_<package> sets variables it
# expects to be exposed in the caller
macro(legate_find_or_configure)
  list(APPEND CMAKE_MESSAGE_CONTEXT "find_or_configure")

  cmake_parse_arguments(_LEGATE_FOC "" "PACKAGE" "" ${ARGN})

  if(NOT _LEGATE_FOC_PACKAGE)
    message(FATAL_ERROR "Must pass PACKAGE argument")
  endif()
  string(TOLOWER "${_LEGATE_FOC_PACKAGE}" _LEGATE_FOC_PACKAGE_LOWER)

  include(
    "${LEGATE_CMAKE_DIR}/thirdparty/get_${_LEGATE_FOC_PACKAGE_LOWER}.cmake" # codespell:ignore thirdparty
    OPTIONAL
    RESULT_VARIABLE _LEGATE_FOC_FOUND
  )

  if(NOT _LEGATE_FOC_FOUND)
    message(FATAL_ERROR "Error getting: ${_LEGATE_FOC_PACKAGE}, no such package")
  endif()

  if(legate_IGNORE_INSTALLED_PACKAGES AND (NOT ${_LEGATE_FOC_PACKAGE}_ROOT))
    message(
      STATUS
      "Ignoring all installed packages when searching for ${_LEGATE_FOC_PACKAGE}"
    )
    # Use set() instead of option() because we definitely want to force this on. option()
    # allows the user to override
    set(CPM_DOWNLOAD_${_LEGATE_FOC_PACKAGE} ON)
    set(
      CPM_DOWNLOAD_${_LEGATE_FOC_PACKAGE}
      ON
      CACHE BOOL
      "Force CPM to download ${_LEGATE_FOC_PACKAGE}"
      FORCE
    )
  endif()

  cmake_language(CALL "find_or_configure_${_LEGATE_FOC_PACKAGE_LOWER}")

  if(NOT LEGATE_SKIP_PATCH_STATUS)
    include("${rapids-cmake-dir}/cpm/detail/display_patch_status.cmake")
    rapids_cpm_display_patch_status("${_LEGATE_FOC_PACKAGE}")
  endif()

  if(${_LEGATE_FOC_PACKAGE}_DIR)
    message(
      STATUS
      "Found external ${_LEGATE_FOC_PACKAGE}_DIR = ${${_LEGATE_FOC_PACKAGE}_DIR}"
    )
  elseif(${_LEGATE_FOC_PACKAGE}_ROOT)
    message(
      STATUS
      "Found external ${_LEGATE_FOC_PACKAGE}_ROOT = ${${_LEGATE_FOC_PACKAGE}_ROOT}"
    )
  else()
    # The following is to head off:
    #
    # 1. ./configure --with-foo (CMake downloads and builds Foo)
    # 2. pip install . (CMake -- as a byproduct of installing legate -- installs Foo)
    # 3. ./reconfigure... (CMake now picks up installed Foo)
    # 4. pip install .
    #
    # In (4), because pip build is smart, it will first uninstall the existing Foo, which
    # also includes the cmake file. But this leads to the Python CMake no longer finding
    # Foo...
    #
    # Setting CPM_DOWNLOAD_<Foo> ensures that CPM converts all CPMFindPackage(Foo) to
    # CPMAddPackage(Foo). We don't want CPMFindPackage() because it calls find_package(),
    # which might find our previously installed binaries.
    #
    # We do need to be careful about using rapids_find_package(), however, since that
    # still calls find_package(). Each of the packages should therefore do:
    #
    # if(NOT CPM_DOWNLOAD_Foo)
    #   rapids_find_package(Foo)
    # else()
    #   rapids_cpm_find(Foo)
    # endif()
    message(
      STATUS
      "${_LEGATE_FOC_PACKAGE}_DIR and ${_LEGATE_FOC_PACKAGE}_ROOT undefined, "
      "forcing CPM to reuse downloaded ${_LEGATE_FOC_PACKAGE} from now on"
    )
    # Use option() here to make this stick in case legate_IGNORE_INSTALLED_PACKAGES wasn't
    # set.
    option(
      CPM_DOWNLOAD_${_LEGATE_FOC_PACKAGE}
      "Force CPM to download ${_LEGATE_FOC_PACKAGE}"
      ON
    )
  endif()

  unset(_LEGATE_FOC_PACKAGE)
  unset(_LEGATE_FOC_PACKAGE_LOWER)
  unset(_LEGATE_FOC_FOUND)
  list(POP_BACK CMAKE_MESSAGE_CONTEXT)
endmacro()

function(legate_maybe_override_package_info package user_branch)
  # CPM_ARGS GIT_TAG and GIT_REPOSITORY don't do anything if you have already overridden
  # those options via a rapids_cpm_package_override() call. So we have to conditionally
  # override the defaults (by creating a temporary json file in build dir) only if the
  # user sets them.

  # See https://github.com/rapidsai/rapids-cmake/issues/575. Specifically, this function
  # is pretty much identical to
  # https://github.com/rapidsai/rapids-cmake/issues/575#issuecomment-2045374410.
  string(TOLOWER "${package}" package_lo)
  cmake_path(
    SET overrides_json
    NORMALIZE
    "${LEGATE_CMAKE_DIR}/versions/${package_lo}_version.json"
  )
  if(user_branch)
    # The user has set either one of these, time to create our cludge.
    file(READ "${overrides_json}" json_data)

    string(JSON old_branch GET "${json_data}" "packages" "${package}" "git_tag")
    if(NOT ("${old_branch}" STREQUAL "${user_branch}"))
      string(
        JSON json_data
        SET "${json_data}"
        "packages"
        "${package}"
        "git_tag"
        "\"${user_branch}\""
      )

      cmake_path(
        SET overrides_json
        NORMALIZE
        "${CMAKE_CURRENT_BINARY_DIR}/${package_lo}_version.json"
      )
      file(WRITE "${overrides_json}" "${json_data}")
    endif()
  endif()
  rapids_cpm_package_override("${overrides_json}")
endfunction()

include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")

function(
  legate_load_overrideable_package_info
  package
  version_var
  git_repo_var
  git_branch_var
  shallow_var
  exclude_from_all_var
)
  rapids_cpm_package_details(
    "${package}"
    version
    git_repo
    git_branch
    shallow
    exclude_from_all
  )
  # https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_cpm_package_override/
  #
  # > Added in version v23.10.00: When the variable CPM_<package_name>_SOURCE exists, any
  # > override entries for package_name will be ignored.
  #
  # This means that our above call to maybe_override Legion might have been completely
  # pointless, and all of the below information is stale. So we have to manually read the
  # override file ourselves.
  if(NOT version)
    if(NOT CPM_${package}_SOURCE)
      # If we don't have a version, and we haven't set the source, then idk why this would
      # fail, but likely the issue isn't on our side
      message(
        FATAL_ERROR
        "rapids-cmake failed to set version information (and likely "
        "all the rest of the fields from the override). Please open a "
        "bug report at https://github.com/rapidsai/rapids-cmake/issues "
        "to report this issue."
      )
    endif()
    string(TOLOWER "${package}" package_lo)
    file(READ "${LEGATE_CMAKE_DIR}/versions/${package_lo}_version.json" json_data)
    string(JSON version GET "${json_data}" "packages" "${package}" "version")
    string(
      JSON shallow
      ERROR_VARIABLE err
      GET "${json_data}"
      "packages"
      "${package}"
      "git_shallow"
    )
    if(err)
      set(shallow FALSE)
    endif()
    string(
      JSON exclude_from_all
      ERROR_VARIABLE err
      GET "${json_data}"
      "packages"
      "${package}"
      "exclude_from_all"
    )
    if(err)
      set(exclude_from_all OFF)
    endif()
  endif()

  set(${version_var} "${version}" PARENT_SCOPE)
  set(${git_repo_var} "${git_repo}" PARENT_SCOPE)
  set(${git_branch_var} "${git_branch}" PARENT_SCOPE)
  set(${shallow_var} "${shallow}" PARENT_SCOPE)
  set(${exclude_from_all_var} "${exclude_from_all}" PARENT_SCOPE)
endfunction()
