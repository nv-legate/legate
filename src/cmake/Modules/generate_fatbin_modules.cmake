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

function(_legate_check_nvcc_pedantic_flags FLAGS)
  if(legate_SKIP_NVCC_PEDANTIC_CHECK)
    message(VERBOSE "Skipping nvcc pedantic check (explicitly skipped by user)")
    return()
  endif()
  if(NOT (CMAKE_CUDA_COMPILER_ID MATCHES "NVIDIA"))
    message(VERBOSE
            "Skipping nvcc pedantic check (compiler \"${CMAKE_CUDA_COMPILER_ID}\" is not nvcc)"
    )
    return()
  endif()
  # We want to catch either "-pedantic" or "--compiler-option=-pedantic" or
  # --compiler-options='-pedantic' but we do NOT want to catch -Wformat-pedantic!
  string(REGEX MATCH [=[[ |=|='|="]\-W?pedantic]=] match_var "${FLAGS}")
  if(match_var)
    message(FATAL_ERROR "-pedantic (or -Wpedantic) is not supported by nvcc and will lead to "
                        "spurious warnings in generated code. Please remove it from your build flags. If "
                        "you would like to override this behavior, reconfigure with "
                        "-Dlegate_SKIP_NVCC_PEDANTIC_CHECK=ON.")
  endif()
endfunction()

function(legate_generate_fatbin_modules)
  list(APPEND CMAKE_MESSAGE_CONTEXT "generate_fatbin_modules")

  set(options)
  set(one_value_args DEST_DIR GENERATED_SOURCES_VAR)
  set(multi_value_args SOURCES EXTRA_FLAGS)
  cmake_parse_arguments(_LEGATE "${options}" "${one_value_args}" "${multi_value_args}"
                        ${ARGN})

  if(NOT _LEGATE_DEST_DIR)
    message(FATAL_ERROR "Must pass DEST_DIR")
  endif()

  if(NOT IS_ABSOLUTE "${_LEGATE_DEST_DIR}")
    set(_LEGATE_DEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/${_LEGATE_DEST_DIR}")
  endif()

  set(cuda_flags ${_LEGATE_EXTRA_FLAGS})

  _legate_check_nvcc_pedantic_flags(cuda_flags)

  include("${LEGATE_CMAKE_DIR}/Modules/utilities.cmake")

  set(src_list)
  set(seen_fatbin_vars)
  foreach(src IN LISTS _LEGATE_SOURCES)
    string(MAKE_C_IDENTIFIER "${src}_fatbin" fatbin_target_name)

    add_library("${fatbin_target_name}" OBJECT "${src}")
    target_include_directories("${fatbin_target_name}"
                               PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
                                       "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/legate"
    )

    target_link_libraries("${fatbin_target_name}"
                          PRIVATE CCCL::CCCL
                                  # Technically none of the remaining libraries need to be
                                  # linked into the fatbins, but since the fatbins include
                                  # legate headers, and therefore transitively include the
                                  # headers from these libraries, we have to include
                                  # them...
                                  Legion::Legion
                                  fmt::fmt-header-only)
    # Don't use cuda_flags for this since it does not handle generator expressions.
    target_compile_options("${fatbin_target_name}"
                           PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:
                                   -Xcudafe=--diag_suppress=boolean_controlling_expr_is_constant
                                   -Xfatbin=-compress-all
                                   --expt-extended-lambda
                                   --expt-relaxed-constexpr
                                   -Wno-deprecated-gpu-targets
                                   --fatbin>)

    set_target_properties("${fatbin_target_name}"
                          PROPERTIES POSITION_INDEPENDENT_CODE ON
                                     INTERFACE_POSITION_INDEPENDENT_CODE ON)

    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.27.0")
      set_target_properties("${fatbin_target_name}" PROPERTIES CUDA_FATBIN_COMPILATION ON)
    endif()

    if(cuda_flags)
      legate_add_target_compile_option("${fatbin_target_name}" CUDA PRIVATE cuda_flags)
    endif()

    cmake_path(GET src STEM fatbin_var_name)
    list_add_if_not_present_error(seen_fatbin_vars "${fatbin_var_name}")
    set(fatbin_cc "${_LEGATE_DEST_DIR}/${fatbin_var_name}.cc")
    set(fatbin_h "${_LEGATE_DEST_DIR}/${fatbin_var_name}.h")
    message(STATUS "Created fatbin target for: ${src}")

    add_custom_command(OUTPUT "${fatbin_cc}"
                       COMMAND ${CMAKE_COMMAND} "-DVAR_NAME=${fatbin_var_name}"
                               "-DIN_FILE=$<TARGET_OBJECTS:${fatbin_target_name}>"
                               "-DOUT_CC_FILE=${fatbin_cc}" "-DOUT_H_FILE=${fatbin_h}"
                               "-DLEGATE_CMAKE_DIR=${LEGATE_CMAKE_DIR}" -P
                               "${LEGATE_CMAKE_DIR}/Modules/bin2c.cmake"
                       VERBATIM
                       WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                       DEPENDS "${fatbin_target_name}"
                       COMMENT "Embedding binary objects $<TARGET_OBJECTS:${fatbin_target_name}> -> ${fatbin_cc}"
    )

    list(APPEND src_list "${fatbin_cc}")
  endforeach()
  set(${_LEGATE_GENERATED_SOURCES_VAR} "${src_list}" PARENT_SCOPE)
endfunction()
