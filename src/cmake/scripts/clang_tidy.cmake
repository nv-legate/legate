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
  # Helper to run one CUDA tidy pass and capture output + statuses
  function(_legate_run_tidy_cuda mode_flag out_var statuses_var)
    execute_process(COMMAND ${CLANG_TIDY} --extra-arg=${mode_flag}
                            --extra-arg=-Wno-unknown-cuda-version -p "${CMAKE_BINARY_DIR}"
                            "${SRC}"
                    COMMAND "${SED}" -E ${_LEGATE_TIDY_SED_RX} #
                    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
                    OUTPUT_VARIABLE _out
                    ERROR_VARIABLE _out RESULTS_VARIABLE _statuses
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)
    set(${out_var} "${_out}" PARENT_SCOPE)
    set(${statuses_var} "${_statuses}" PARENT_SCOPE)
  endfunction()

  # Host-only
  _legate_run_tidy_cuda("--cuda-host-only" output_host statuses_host)
  list(GET statuses_host 0 clang_tidy_status_host)
  list(GET statuses_host 1 sed_status_host)

  # Device-only
  _legate_run_tidy_cuda("--cuda-device-only" output_device statuses_device)
  list(GET statuses_device 0 clang_tidy_status_device)
  list(GET statuses_device 1 sed_status_device)

  # Combine outputs and statuses
  set(output "${output_host}\n${output_device}")
  if(clang_tidy_status_host OR clang_tidy_status_device)
    set(clang_tidy_status 1)
  endif()
  if(sed_status_host OR sed_status_device)
    set(sed_status 1)
  endif()
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
