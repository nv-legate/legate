#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

function(rapids_cmake_build_type default_type)
endfunction()

function(rapids_cmake_support_conda_env target)
  set(options MODIFY_PREFIX_PATH)
  set(one_value_args)
  set(multi_value_args)

  cmake_parse_arguments("${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
endfunction()

function(rapids_cmake_install_lib_dir out_variable_name)
  set(options MODIFY_INSTALL_LIBDIR)
  set(one_value_args)
  set(multi_value_args)

  cmake_parse_arguments("${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
endfunction()

function(rapids_cmake_write_version_file file_path)
  set(options)
  set(one_value PREFIX)
  set(multi_value)

  cmake_parse_arguments(_RAPIDS "${options}" "${one_value}" "${multi_value}" ${ARGN})
endfunction()

function(rapids_cpm_init)
  set(_rapids_options GENERATE_PINNED_VERSIONS)
  set(_rapids_one_value CUSTOM_DEFAULT_VERSION_FILE OVERRIDE)
  set(_rapids_multi_value)

  cmake_parse_arguments(
    _RAPIDS
    "${_rapids_options}"
    "${_rapids_one_value}"
    "${_rapids_multi_value}"
    ${ARGN}
  )
endfunction()

function(rapids_export type project_name)
  set(options)
  set(
    one_value
    EXPORT_SET
    VERSION
    NAMESPACE
    DOCUMENTATION
    FINAL_CODE_BLOCK
  )
  set(multi_value GLOBAL_TARGETS COMPONENTS COMPONENTS_EXPORT_SET LANGUAGES)

  cmake_parse_arguments(_RAPIDS "${options}" "${one_value}" "${multi_value}" ${ARGN})
endfunction()

function(rapids_cython_init)
endfunction()

function(rapids_cython_create_modules)
  set(_rapids_cython_options CXX)
  set(_rapids_cython_one_value INSTALL_DIR MODULE_PREFIX COMPONENT)
  set(_rapids_cython_multi_value SOURCE_FILES LINKED_LIBRARIES ASSOCIATED_TARGETS)

  cmake_parse_arguments(
    _RAPIDS_CYTHON
    "${_rapids_cython_options}"
    "${_rapids_cython_one_value}"
    "${_rapids_cython_multi_value}"
    ${ARGN}
  )
endfunction()

function(rapids_test_init)
endfunction()

function(rapids_test_add)
  set(options)
  set(
    one_value
    NAME
    WORKING_DIRECTORY
    GPUS
    PERCENT
    INSTALL_COMPONENT_SET
    INSTALL_TARGET
  )
  set(multi_value COMMAND)

  cmake_parse_arguments(_RAPIDS_TEST "${options}" "${one_value}" "${multi_value}" ${ARGN})
endfunction()

function(rapids_test_install_relocatable)
  set(options INCLUDE_IN_ALL)
  set(one_value INSTALL_COMPONENT_SET DESTINATION)
  set(multi_value)

  cmake_parse_arguments(_RAPIDS_TEST "${options}" "${one_value}" "${multi_value}" ${ARGN})
endfunction()

function(rapids_test_gpu_requirements)
  set(options)
  set(one_value GPUS PERCENT)
  set(multi_value)

  cmake_parse_arguments(_RAPIDS_TEST "${options}" "${one_value}" "${multi_value}" ${ARGN})
endfunction()

function(
  rapids_cpm_package_details
  package_name
  version_var
  url_var
  tag_var
  shallow_var
  exclude_from_all_var
)
endfunction()

function(rapids_cpm_cccl)
  set(options)
  set(one_value BUILD_EXPORT_SET INSTALL_EXPORT_SET)
  set(multi_value)

  cmake_parse_arguments(_RAPIDS "${options}" "${one_value}" "${multi_value}" ${ARGN})
endfunction()

function(rapids_cpm_gbench)
  set(options BUILD_STATIC)
  set(one_value BUILD_EXPORT_SET INSTALL_EXPORT_SET)
  set(multi_value)

  cmake_parse_arguments(_RAPIDS "${options}" "${one_value}" "${multi_value}" ${ARGN})
endfunction()

function(rapids_cpm_fmt)
  set(options)
  set(one_value BUILD_EXPORT_SET INSTALL_EXPORT_SET)
  set(multi_value)

  cmake_parse_arguments(_RAPIDS "${options}" "${one_value}" "${multi_value}" ${ARGN})
endfunction()

function(rapids_cpm_gtest)
  set(options BUILD_STATIC)
  set(one_value BUILD_EXPORT_SET INSTALL_EXPORT_SET)
  set(multi_value)

  cmake_parse_arguments(_RAPIDS "${options}" "${one_value}" "${multi_value}" ${ARGN})
endfunction()

function(rapids_find_generate_module name)
  set(options NO_CONFIG)
  set(
    one_value
    VERSION
    BUILD_EXPORT_SET
    INSTALL_EXPORT_SET
    INITIAL_CODE_BLOCK
    FINAL_CODE_BLOCK
  )
  set(multi_value HEADER_NAMES LIBRARY_NAMES INCLUDE_SUFFIXES)

  cmake_parse_arguments(_RAPIDS "${options}" "${one_value}" "${multi_value}" ${ARGN})
endfunction()

function(rapids_export_parse_version rapids_version orig_prefix ver_value)
endfunction()

function(rapids_cython_add_rpath_entries)
  set(options)
  set(one_value ROOT_DIRECTORY TARGET)
  set(multi_value PATHS)

  cmake_parse_arguments(
    _RAPIDS_CYTHON
    "${options}"
    "${one_value}"
    "${multi_value}"
    ${ARGN}
  )
endfunction()

function(rapids_cpm_display_patch_status package_name)
endfunction()

function(rapids_cpm_package_override _rapids_override_filepath)
endfunction()
