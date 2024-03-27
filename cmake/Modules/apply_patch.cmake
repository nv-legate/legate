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

function(legate_core_generate_patch_command)
  list(APPEND CMAKE_MESSAGE_CONTEXT "generate_patch_command")

  set(options)
  set(one_value_args SOURCE PATCH_FILE DEST_VAR)
  set(multi_value_keywords)
  cmake_parse_arguments(_LEGATE_CORE_GEN_PATCH_CMD "${options}" "${one_value_args}" "${multi_value_keywords}" ${ARGN})

  if(NOT _LEGATE_CORE_GEN_PATCH_CMD_SOURCE)
    message(FATAL_ERROR "No SOURCE argument given to legate_core_generate_patch_command")
  else()
    cmake_path(SET input_file NORMALIZE "${_LEGATE_CORE_GEN_PATCH_CMD_SOURCE}")
  endif()

  if(NOT _LEGATE_CORE_GEN_PATCH_CMD_PATCH_FILE)
    message(FATAL_ERROR "No PATCH_FILE argument given to legate_core_generate_patch_command")
  else()
    cmake_path(SET patch_file NORMALIZE "${_LEGATE_CORE_GEN_PATCH_CMD_PATCH_FILE}")
  endif()

  if(NOT _LEGATE_CORE_GEN_PATCH_CMD_DEST_VAR)
    message(FATAL_ERROR "No DEST_VAR argument given to legate_core_generate_patch_command")
  endif()

  find_package(Patch REQUIRED)

  set(
    ${_LEGATE_CORE_GEN_PATCH_CMD_DEST_VAR}
    ${Patch_EXECUTABLE} ${input_file} -N --input=${patch_file} --ignore-whitespace --quiet
    PARENT_SCOPE
  )
endfunction()

function(legate_core_apply_patch)
  list(APPEND CMAKE_MESSAGE_CONTEXT "apply_patch")

  set(options)
  set(one_value_args SOURCE PATCH_FILE)
  set(multi_value_keywords)
  cmake_parse_arguments(_LEGATE_CORE_PATCH "${options}" "${one_value_args}" "${multi_value_keywords}" ${ARGN})

  legate_core_generate_patch_command(
    SOURCE     ${_LEGATE_CORE_PATCH_SOURCE}
    PATCH_FILE ${_LEGATE_CORE_PATCH_PATCH_FILE}
    DEST_VAR   patch_command
  )

  message(STATUS "Patching ${_LEGATE_CORE_PATCH_SOURCE}")

  execute_process(
    COMMAND ${patch_command}
    TIMEOUT 15
    COMMAND_ERROR_IS_FATAL ANY
  )
endfunction()
