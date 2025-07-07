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

execute_process(COMMAND ${CLANG_TIDY} -p "${CMAKE_BINARY_DIR}" "${SRC}"
                COMMAND "${SED}" -E [=[/[0-9]+ warnings generated\./d]=] #
                WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
                OUTPUT_VARIABLE output
                ERROR_VARIABLE output RESULTS_VARIABLE statuses
                OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)

list(GET statuses 0 clang_tidy_status)
list(GET statuses 1 sed_status)
if(clang_tidy_status OR sed_status)
  message("${output}")
  message("clang-tidy return-code: ${clang_tidy_status}")
  message("sed return-code: ${sed_status}")
  cmake_language(EXIT 1)
endif()
