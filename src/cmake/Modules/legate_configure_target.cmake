#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

function(legate_configure_target _LEGATE_TARGET)
  list(APPEND CMAKE_MESSAGE_CONTEXT "legate_configure_target")

  set(options)
  set(one_value_args)
  set(multi_value_args)
  cmake_parse_arguments(_LEGATE "${options}" "${one_value_args}" "${multi_value_args}"
                        ${ARGN})

  include(GNUInstallDirs)

  set_target_properties("${_LEGATE_TARGET}"
                        PROPERTIES POSITION_INDEPENDENT_CODE ON
                                   RUNTIME_OUTPUT_DIRECTORY "${CMAKE_INSTALL_BINDIR}"
                                   LIBRARY_OUTPUT_DIRECTORY "${CMAKE_INSTALL_LIBDIR}"
                                   ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_INSTALL_LIBDIR}"
                                   LEGATE_INTERNAL_TARGET TRUE
                                   CXX_STANDARD 17
                                   CXX_STANDARD_REQUIRED ON)

  target_link_libraries("${_LEGATE_TARGET}" PUBLIC legate::legate)
endfunction()
