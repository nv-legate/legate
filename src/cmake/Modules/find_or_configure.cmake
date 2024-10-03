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

function(legate_parse_versions_json)
  list(APPEND CMAKE_MESSAGE_CONTEXT "parse_versions_json")

  set(options)
  set(one_value_args PACKAGE VERSION GIT_URL GIT_TAG GIT_SHALLOW)
  set(multi_value_args)
  cmake_parse_arguments(_LEGATE_PVJ "${options}" "${one_value_args}"
                        "${multi_value_args}" ${ARGN})

  if(_LEGATE_PVJ_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unparsed arguments: ${_LEGATE_PVJ_UNPARSED_ARGUMENTS}")
  endif()

  if(_LEGATE_PVJ_KEYWORDS_MISSING_VALUES)
    message(FATAL_ERROR "${_LEGATE_PVJ_KEYWORDS_MISSING_VALUES}")
  endif()

  if(NOT _LEGATE_PVJ_PACKAGE)
    message(FATAL_ERROR "Must pass PACKAGE")
  endif()

  file(READ "${legate_VERSIONS_JSON}" versions_json)
  string(JSON mdspan_json GET "${versions_json}" "packages" "${_LEGATE_PVJ_PACKAGE}")

  if(_LEGATE_PVJ_VERSION)
    string(JSON version GET "${mdspan_json}" "version")
    set(${_LEGATE_PVJ_VERSION} "${version}" PARENT_SCOPE)
  endif()

  if(_LEGATE_PVJ_GIT_URL)
    string(JSON git_url GET "${mdspan_json}" "git_url")
    set(${_LEGATE_PVJ_GIT_URL} "${git_url}" PARENT_SCOPE)
  endif()

  if(_LEGATE_PVJ_GIT_TAG)
    string(JSON git_tag GET "${mdspan_json}" "git_tag")
    set(${_LEGATE_PVJ_GIT_TAG} "${git_tag}" PARENT_SCOPE)
  endif()

  if(_LEGATE_PVJ_GIT_SHALLOW)
    string(JSON git_shallow GET "${mdspan_json}" "git_shallow")

    if(git_shallow STREQUAL "ON")
      set(git_shallow TRUE)
    else()
      if(NOT (git_shallow STREQUAL "OFF"))
        message(FATAL_ERROR "git_shallow unexpected value: ${git_shallow}")
      endif()
      set(git_shallow FALSE)
    endif()

    set(${_LEGATE_PVJ_GIT_SHALLOW} "${git_shallow}" PARENT_SCOPE)
  endif()
endfunction()

# This guy needs to be a macro in case the find_or_configure_<package> sets variables it
# expects to be exposed in the caller
macro(legate_find_or_configure)
  list(APPEND CMAKE_MESSAGE_CONTEXT "find_or_configure")

  cmake_parse_arguments(_LEGATE_FOC "" "PACKAGE" "" ${ARGN})

  if(NOT _LEGATE_FOC_PACKAGE)
    message(FATAL_ERROR "Must pass PACKAGE argument")
  endif()
  string(TOLOWER "${_LEGATE_FOC_PACKAGE}" _LEGATE_FOC_PACKAGE_LOWER)

  include("${LEGATE_CMAKE_DIR}/thirdparty/get_${_LEGATE_FOC_PACKAGE_LOWER}.cmake" OPTIONAL
          RESULT_VARIABLE _LEGATE_FOC_FOUND)

  if(NOT _LEGATE_FOC_FOUND)
    message(FATAL_ERROR "Error getting: ${_LEGATE_FOC_PACKAGE}, no such package")
  endif()

  if(legate_IGNORE_INSTALLED_PACKAGES AND (NOT ${_LEGATE_FOC_PACKAGE}_ROOT))
    message(STATUS "Ignoring all installed packages when searching for ${_LEGATE_FOC_PACKAGE}"
    )
    set(CPM_DOWNLOAD_${_LEGATE_FOC_PACKAGE} ON CACHE BOOL "" FORCE)
  endif()

  cmake_language(CALL "find_or_configure_${_LEGATE_FOC_PACKAGE_LOWER}")

  if(${_LEGATE_FOC_PACKAGE}_DIR)
    message(STATUS "Found external ${_LEGATE_FOC_PACKAGE}_DIR = ${${_LEGATE_FOC_PACKAGE}_DIR}"
    )
  elseif(${_LEGATE_FOC_PACKAGE}_ROOT)
    message(STATUS "Found external ${_LEGATE_FOC_PACKAGE}_ROOT = ${${_LEGATE_FOC_PACKAGE}_ROOT}"
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
    # if(NOT CPM_DOWNLOAD_Foo AND NOT CPM_Foo_SOURCE) rapids_find_package(Foo) else()
    # rapids_cpm_find(Foo) endif()
    message(STATUS "${_LEGATE_FOC_PACKAGE}_DIR and ${_LEGATE_FOC_PACKAGE}_ROOT undefined, "
                   "forcing CPM to re-use downloaded ${_LEGATE_FOC_PACKAGE} from now on")
    set(CPM_DOWNLOAD_${_LEGATE_FOC_PACKAGE} ON CACHE BOOL "" FORCE)
    if(CPM_PACKAGE_${_LEGATE_FOC_PACKAGE}_SOURCE_DIR)
      # If the local package path was supplied by the user, this will be populated to the
      # right place.
      set(CPM_${_LEGATE_FOC_PACKAGE}_SOURCE
          "${CPM_PACKAGE_${_LEGATE_FOC_PACKAGE}_SOURCE_DIR}" CACHE PATH "" FORCE)
    else()
      set(CPM_${_LEGATE_FOC_PACKAGE}_SOURCE
          "${FETCHCONTENT_BASE_DIR}/${_LEGATE_FOC_PACKAGE_LOWER}-src" CACHE PATH "" FORCE)
    endif()

  endif()

  unset(_LEGATE_FOC_PACKAGE)
  unset(_LEGATE_FOC_PACKAGE_LOWER)
  unset(_LEGATE_FOC_FOUND)
  list(POP_BACK CMAKE_MESSAGE_CONTEXT)
endmacro()
