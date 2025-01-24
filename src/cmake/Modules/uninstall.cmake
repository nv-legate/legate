#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

function(legate_uninstall_target)
  list(APPEND CMAKE_MESSAGE_CONTEXT "uninstall_target")

  set(options)
  set(one_value_args TARGET)
  set(multi_value_args)

  cmake_parse_arguments(_LEGATE "${options}" "${one_value_args}" "${multi_value_args}"
                        ${ARGN})

  if(NOT _LEGATE_TARGET)
    message(FATAL_ERROR "Must pass TARGET")
  endif()

  if(TARGET ${_LEGATE_TARGET})
    # Nothing to do
    message(STATUS "Uninstall target '${_LEGATE_TARGET}' already exists")
    return()
  endif()

  set(INSTALL_MANIFEST_PATH "${CMAKE_BINARY_DIR}/install_manifest.txt")

  configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../templates/uninstall.cmake.in"
                 "${CMAKE_CURRENT_BINARY_DIR}/uninstall.cmake" @ONLY)

  add_custom_target("${_LEGATE_TARGET}"
                    COMMAND "${CMAKE_COMMAND}" -P
                            "${CMAKE_CURRENT_BINARY_DIR}/uninstall.cmake"
                    COMMENT "Uninstalling ${CMAKE_PROJECT_NAME}")
endfunction()
