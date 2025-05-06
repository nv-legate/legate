#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

foreach(var ROOT_DIR BUILD_DIR CLANG_TIDY SED SOURCES)
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

execute_process(COMMAND ${CMAKE_COMMAND} -S "${ROOT_DIR}" -B "${BUILD_DIR}"
                        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON --fresh
                WORKING_DIRECTORY "${ROOT_DIR}"
                OUTPUT_VARIABLE output
                ERROR_VARIABLE output
                RESULT_VARIABLE return_code)

if(return_code)
  message(FATAL_ERROR "Error building external tidy target:\n${output}")
endif()

separate_arguments(CLANG_TIDY) # cmake-lint: disable=E1120

foreach(src IN LISTS SOURCES)
  execute_process(COMMAND ${CLANG_TIDY} -p "${BUILD_DIR}" "${src}"
                  COMMAND "${SED}" -E [=[/[0-9]+ warnings generated\./d]=] #
                  WORKING_DIRECTORY "${BUILD_DIR}"
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
endforeach()
