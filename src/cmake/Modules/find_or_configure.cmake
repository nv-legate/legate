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
    #
    # cmake-format: off
    install(
      TARGETS "${target}"
      ARCHIVE
        DESTINATION "${legate_DEP_INSTALL_LIBDIR}"
        ${archive_namelink_skip}
      LIBRARY
        DESTINATION "${legate_DEP_INSTALL_LIBDIR}"
        NAMELINK_SKIP
      RUNTIME
        DESTINATION "${legate_DEP_INSTALL_BINDIR}"
      PUBLIC_HEADER
        DESTINATION "${legate_DEP_INSTALL_INCLUDEDIR}"
      PRIVATE_HEADER
        DESTINATION "${legate_DEP_INSTALL_INCLUDEDIR}"
      INCLUDES
        DESTINATION "${legate_DEP_INSTALL_INCLUDEDIR}")
    # cmake-format: on

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
  foreach(suffix MAJOR MINOR PATCH TWEAK COUNT)
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

  # cmake-format: off
  include(
    "${LEGATE_CMAKE_DIR}/thirdparty/get_${_LEGATE_FOC_PACKAGE_LOWER}.cmake" # codespell:ignore thirdparty
    OPTIONAL
    RESULT_VARIABLE _LEGATE_FOC_FOUND)
  # cmake-format: on

  if(NOT _LEGATE_FOC_FOUND)
    message(FATAL_ERROR "Error getting: ${_LEGATE_FOC_PACKAGE}, no such package")
  endif()

  if(legate_IGNORE_INSTALLED_PACKAGES AND (NOT ${_LEGATE_FOC_PACKAGE}_ROOT))
    message(STATUS "Ignoring all installed packages when searching for ${_LEGATE_FOC_PACKAGE}"
    )
    # Use set() instead of option() because we definitely want to force this on. option()
    # allows the user to override
    set(CPM_DOWNLOAD_${_LEGATE_FOC_PACKAGE} ON)
    set(CPM_DOWNLOAD_${_LEGATE_FOC_PACKAGE} ON
        CACHE BOOL "Force CPM to download ${_LEGATE_FOC_PACKAGE}" FORCE)
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
    # cmake-format: off
    # if(NOT CPM_DOWNLOAD_Foo)
    #   rapids_find_package(Foo)
    # else()
    #   rapids_cpm_find(Foo)
    # endif()
    # cmake-format: on
    message(STATUS "${_LEGATE_FOC_PACKAGE}_DIR and ${_LEGATE_FOC_PACKAGE}_ROOT undefined, "
                   "forcing CPM to reuse downloaded ${_LEGATE_FOC_PACKAGE} from now on")
    # Use option() here to make this stick in case legate_IGNORE_INSTALLED_PACKAGES wasn't
    # set.
    option(CPM_DOWNLOAD_${_LEGATE_FOC_PACKAGE}
           "Force CPM to download ${_LEGATE_FOC_PACKAGE}" ON)
  endif()

  unset(_LEGATE_FOC_PACKAGE)
  unset(_LEGATE_FOC_PACKAGE_LOWER)
  unset(_LEGATE_FOC_FOUND)
  list(POP_BACK CMAKE_MESSAGE_CONTEXT)
endmacro()
