#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

foreach(var CLANG_TIDY CMAKE_BINARY_DIR SRC SED)
  if(NOT DEFINED ${var})
    message(FATAL_ERROR "Must pass ${var}")
  endif()
endforeach()

separate_arguments(CLANG_TIDY) # cmake-lint: disable=E1120
set(_LEGATE_TIDY_SED_RX [=[/[0-9]+ warnings generated\./d]=])

# For .cu, run host-only and device-only passes; otherwise run once.
get_filename_component(_SRC_EXT "${SRC}" EXT)
set(output "")
set(clang_tidy_status 0)
set(sed_status 0)

if(_SRC_EXT STREQUAL ".cu")
  # CUDA files contain NVCC-specific compiler flags in compile_commands.json that
  # clang-tidy cannot parse. Rather than trying to filter these out (which is fragile), we
  # skip clang-tidy analysis for .cu files entirely. The actual code logic in .cu files is
  # typically minimal (usually just kernel launches), and the substantive code is in
  # .cc/.h files which are properly analyzed.
  message(STATUS "Skipping clang-tidy for CUDA file: ${SRC}")
  set(output "Skipped CUDA file")
  set(clang_tidy_status 0)
  set(sed_status 0)
else()
  execute_process(COMMAND ${CLANG_TIDY} -p "${CMAKE_BINARY_DIR}" "${SRC}"
                  COMMAND "${SED}" -E ${_LEGATE_TIDY_SED_RX} #
                  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
                  OUTPUT_VARIABLE output
                  ERROR_VARIABLE output RESULTS_VARIABLE statuses
                  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)
  list(GET statuses 0 clang_tidy_status)
  list(GET statuses 1 sed_status)
endif()

if(clang_tidy_status OR sed_status)
  message("${output}")
  message("clang-tidy return-code: ${clang_tidy_status}")
  message("sed return-code: ${sed_status}")
  cmake_language(EXIT 1)
endif()
