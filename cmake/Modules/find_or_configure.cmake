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

# This guy needs to be a macro in case the find_or_configure_<package> sets variables it
# expects to be exposed in the caller
macro(legate_core_find_or_configure)
  list(APPEND CMAKE_MESSAGE_CONTEXT "find_or_configure")

  cmake_parse_arguments(_LEGATE_CORE_FOC "" "PACKAGE" "" ${ARGN})

  if(NOT _LEGATE_CORE_FOC_PACKAGE)
    message(FATAL_ERROR "Must pass PACKAGE argument")
  endif()
  string(TOLOWER "${_LEGATE_CORE_FOC_PACKAGE}" _LEGATE_CORE_FOC_PACKAGE_LOWER)

  include(
    ${LEGATE_CORE_DIR}/cmake/thirdparty/get_${_LEGATE_CORE_FOC_PACKAGE_LOWER}.cmake
    OPTIONAL
    RESULT_VARIABLE _LEGATE_CORE_FOC_FOUND
  )

  if(NOT _LEGATE_CORE_FOC_FOUND)
    message(FATAL_ERROR "Error getting: ${_LEGATE_CORE_FOC_PACKAGE}, no such package")
  endif()

  cmake_language(CALL "find_or_configure_${_LEGATE_CORE_FOC_PACKAGE_LOWER}")

  if((NOT ${_LEGATE_CORE_FOC_PACKAGE}_DIR) AND (NOT ${_LEGATE_CORE_FOC_PACKAGE}_ROOT))
    # The following is to head off:
    #
    # 1. ./configure --with-foo (CMake downloads and builds Foo)
    # 2. pip install . (CMake -- as a byproduct of installing legate.core -- installs Foo)
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
    # if(NOT CPM_DOWNLOAD_Foo AND NOT CPM_Foo_SOURCE)
    #   rapids_find_package(Foo)
    # else()
    #    rapids_cpm_find(Foo)
    # endif()
    set(CPM_DOWNLOAD_${_LEGATE_CORE_FOC_PACKAGE} ON CACHE BOOL "" FORCE)
    set(
      CPM_${_LEGATE_CORE_FOC_PACKAGE}_SOURCE "${FETCHCONTENT_BASE_DIR}/${_LEGATE_CORE_FOC_PACKAGE_LOWER}-src"
      CACHE PATH "" FORCE
    )
  endif()

  unset(_LEGATE_CORE_FOC_PACKAGE)
  unset(_LEGATE_CORE_FOC_PACKAGE_LOWER)
  unset(_LEGATE_CORE_FOC_FOUND)
  list(POP_BACK CMAKE_MESSAGE_CONTEXT)
endmacro()
