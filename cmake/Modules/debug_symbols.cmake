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

function(_legate_core_debug_syms_macos target install_dir)
  list(APPEND CMAKE_MESSAGE_CONTEXT "macos")

  # Both clang and gcc will automatically generate a <TARGET>.dSYM directory (the debug
  # symbols) when creating single executables, but refuse to do so when creating
  # libraries. So we must do this ourselves...
  find_program(LEGATE_CORE_DSYMUTIL dsymutil)
  if (LEGATE_CORE_DSYMUTIL)
    add_custom_command(
      TARGET ${target} POST_BUILD
      COMMAND "${LEGATE_CORE_DSYMUTIL}" "$<TARGET_FILE_NAME:${target}>"
      WORKING_DIRECTORY "$<TARGET_FILE_DIR:${target}>"
      DEPENDS ${target}
    )

    # We want to install the dsymutil stuff directly next to the installed
    # binary/lib. Unfortunately, there is no way to query where that path is (CMake simply
    # doesn't keep track). We could write it in ourselves by setting a custom property,
    # but then people will forget to do that and this breaks.
    #
    # So instead we need them to tell us exactly where to put it...
    install(
      DIRECTORY "$<TARGET_FILE:${target}>.dSYM"
      DESTINATION ${install_dir}
    )
  endif()
endfunction()

function(legate_core_debug_syms target)
  list(APPEND CMAKE_MESSAGE_CONTEXT "debug_syms")

  set(options)
  set(one_value_args INSTALL_DIR)
  set(multi_value_keywords)
  cmake_parse_arguments(_DEBUG_SYMS "${options}" "${one_value_args}" "${multi_value_keywords}" ${ARGN})

  if (NOT _DEBUG_SYMS_INSTALL_DIR)
    message(FATAL_ERROR "Must pass INSTALL_DIR")
  endif()
  if ((CMAKE_BUILD_TYPE STREQUAL "Debug") OR (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo"))
    if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
      _legate_core_debug_syms_macos(${target} ${_DEBUG_SYMS_INSTALL_DIR})
    endif()
    # nothing to do for other OS's for now
  endif()
endfunction()
