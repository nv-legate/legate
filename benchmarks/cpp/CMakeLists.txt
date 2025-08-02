#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

list(APPEND CMAKE_MESSAGE_CONTEXT "cpp")

include("${LEGATE_CMAKE_DIR}/Modules/debug_symbols.cmake")
include("${LEGATE_CMAKE_DIR}/Modules/clang_tidy.cmake")

legate_find_or_configure(PACKAGE google_benchmark)

function(legate_configure_benchmark)
  set(options)
  set(one_value TARGET)
  set(multi_value SOURCES)
  cmake_parse_arguments(_LEGATE_BM "${options}" "${one_value}" "${multi_value}" ${ARGN})

  add_executable(${_LEGATE_BM_TARGET} ${_LEGATE_BM_SOURCES})

  set_target_properties(${_LEGATE_BM_TARGET}
                        PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                   "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}>"
                                   BUILD_RPATH_USE_ORIGIN TRUE
                                   INSTALL_RPATH_USE_LINK_PATH TRUE
                                   LEGATE_INTERNAL_TARGET TRUE)

  set_property(TARGET ${_LEGATE_BM_TARGET} APPEND
               PROPERTY INSTALL_RPATH
                        "${legate_PLATFORM_RPATH_ORIGIN}/../${CMAKE_INSTALL_LIBDIR}")

  target_link_libraries(${_LEGATE_BM_TARGET} PRIVATE legate::legate benchmark::benchmark)

  if(legate_USE_CUDA)
    target_link_libraries(${_LEGATE_BM_TARGET} PRIVATE NCCL::NCCL)
  endif()

  legate_install_debug_symbols(TARGET ${_LEGATE_BM_TARGET}
                               INSTALL_DIR "${CMAKE_INSTALL_BINDIR}")

  foreach(src IN LISTS _LEGATE_BM_SOURCES)
    legate_add_tidy_target(SOURCE "${src}")
  endforeach()
endfunction()

legate_configure_benchmark(TARGET inline_launch SOURCES inline_launch.cc)
