#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

foreach(
  var
  ROOT_DIR
  BUILD_DIR
  CLANG_TIDY
  SED
  SOURCES
  LEGATE_BUILD_DIR
)
  if(NOT DEFINED ${var})
    message(FATAL_ERROR "Must pass ${var}")
  endif()
endforeach()

if(NOT IS_DIRECTORY "${ROOT_DIR}")
  message(FATAL_ERROR "Root directory ${ROOT_DIR} is not readable, or does not exist")
endif()

if(NOT IS_DIRECTORY "${BUILD_DIR}")
  # We could just handle this by transparently creating the directory, but if the build
  # dir doesn't exist, this implies a bug in our original setup code for this target. And
  # since file(CREATE) will create the *whole* tree if any parent dirs are missing, we'd
  # rather not accidentally create that.
  message(FATAL_ERROR "Build directory ${BUILD_DIR} is not readable, or does not exist")
endif()

if(NOT SOURCES)
  message(FATAL_ERROR "Must pass non-empty source list (have ${SOURCES})")
endif()

# Legion is not found by CPMFindPackage in legeate-dependencies.cmake as Legion is trying
# to find Realm. So we need to manually set the CPM_Legion_SOURCE and some Legion flags.
execute_process(
  COMMAND
    ${CMAKE_COMMAND} -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    --fresh -Dlegate_DIR=${LEGATE_BUILD_DIR}
  WORKING_DIRECTORY "${ROOT_DIR}"
  OUTPUT_VARIABLE output
  ERROR_VARIABLE output
  RESULT_VARIABLE return_code
)

if(return_code)
  message(FATAL_ERROR "Error building external tidy target:\n${output}")
endif()

separate_arguments(CLANG_TIDY)
set(_LEGATE_TIDY_SED_RX [=[/[0-9]+ warnings generated\./d]=])

foreach(src IN LISTS SOURCES)
  get_filename_component(_SRC_EXT "${src}" EXT)
  set(output "")
  set(clang_tidy_status 0)
  set(sed_status 0)

  if(_SRC_EXT STREQUAL ".cu")
    # Helper for CUDA host/device-only passes
    function(_ext_run_tidy_cuda mode_flag out_var statuses_var)
      execute_process(
        COMMAND ${CLANG_TIDY} --extra-arg=${mode_flag} -p "${BUILD_DIR}" "${src}"
        COMMAND
          "${SED}" -E ${_LEGATE_TIDY_SED_RX} #
        WORKING_DIRECTORY "${BUILD_DIR}"
        OUTPUT_VARIABLE _out
        ERROR_VARIABLE _out
        RESULTS_VARIABLE _statuses
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
      )
      set(${out_var} "${_out}" PARENT_SCOPE)
      set(${statuses_var} "${_statuses}" PARENT_SCOPE)
    endfunction()

    _ext_run_tidy_cuda("--cuda-host-only" output_host statuses_host)
    list(GET statuses_host 0 clang_tidy_status_host)
    list(GET statuses_host 1 sed_status_host)

    _ext_run_tidy_cuda("--cuda-device-only" output_device statuses_device)
    list(GET statuses_device 0 clang_tidy_status_device)
    list(GET statuses_device 1 sed_status_device)

    set(output "${output_host}\n${output_device}")
    if(clang_tidy_status_host OR clang_tidy_status_device)
      set(clang_tidy_status 1)
    endif()
    if(sed_status_host OR sed_status_device)
      set(sed_status 1)
    endif()
  else()
    execute_process(
      COMMAND ${CLANG_TIDY} -p "${BUILD_DIR}" "${src}"
      COMMAND
        "${SED}" -E ${_LEGATE_TIDY_SED_RX} #
      WORKING_DIRECTORY "${BUILD_DIR}"
      OUTPUT_VARIABLE output
      ERROR_VARIABLE output
      RESULTS_VARIABLE statuses
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_STRIP_TRAILING_WHITESPACE
    )
    list(GET statuses 0 clang_tidy_status)
    list(GET statuses 1 sed_status)
  endif()

  if(clang_tidy_status OR sed_status)
    message("${output}")
    message("clang-tidy return-code: ${clang_tidy_status}")
    message("sed return-code: ${sed_status}")
    cmake_language(EXIT 1)
  endif()
endforeach()
