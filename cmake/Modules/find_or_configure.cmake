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
    set(
      ${_LEGATE_CORE_FOC_PACKAGE}_DIR
      "${FETCHCONTENT_BASE_DIR}/${_LEGATE_CORE_FOC_PACKAGE_LOWER}-build"
      CACHE PATH "" FORCE
    )
  endif()

  unset(_LEGATE_CORE_FOC_PACKAGE)
  unset(_LEGATE_CORE_FOC_PACKAGE_LOWER)
  unset(_LEGATE_CORE_FOC_FOUND)
  list(POP_BACK CMAKE_MESSAGE_CONTEXT)
endmacro()
