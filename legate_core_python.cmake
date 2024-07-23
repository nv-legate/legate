#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

list(APPEND CMAKE_MESSAGE_CONTEXT "python")

# ########################################################################################
# * conda environment --------------------------------------------------------

include(GNUInstallDirs)

rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)

# We're building python extension libraries, which must always be installed under lib/,
# even if the system normally uses lib64/. Rapids-cmake currently doesn't realize this
# when we're going through scikit-build, see
# https://github.com/rapidsai/rapids-cmake/issues/426
if(TARGET conda_env)
  set(CMAKE_INSTALL_LIBDIR "lib")
endif()

rapids_cmake_install_lib_dir(lib_dir)

if(legate_core_SETUP_PY_MODE STREQUAL "develop")
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
endif()

# ########################################################################################
# * Find or build the C++ core -----------------------------------------------

include(cmake/Modules/find_legate_core_cpp.cmake)

find_legate_core_cpp(VERSION ${legate_core_version} BUILD_EXPORT_SET
                     legate-core-python-exports INSTALL_EXPORT_SET
                     legate-core-python-exports)

# ########################################################################################
# * Cython build -------------------------------------------------------------

rapids_find_package(Python3
                    GLOBAL_TARGETS Python3::Module
                    BUILD_EXPORT_SET legate-core-python-exports
                    INSTALL_EXPORT_SET legate-core-python-exports
                    FIND_ARGS
                    REQUIRED
                    COMPONENTS Development)

# For scikit-build. They use some deprecated FindPython() modules that are removed by
# default in cmake 3.28. Setting this policy to OLD restores them.
if(POLICY CMP0148)
  cmake_policy(PUSH)
  cmake_policy(SET CMP0148 OLD)
endif()

include(rapids-cython)

rapids_cython_init()

if(POLICY CMP0148)
  cmake_policy(POP)
endif()

add_library(legate_core_python INTERFACE)
add_library(legate::core_python ALIAS legate_core_python)
target_link_libraries(legate_core_python INTERFACE legate::core)

add_subdirectory(legate)

include(cmake/Modules/cython_rpaths.cmake)

legate_core_populate_cython_dependency_rpaths(RESULT_VAR legate_cython_rpaths)

rapids_cython_add_rpath_entries(TARGET legate::core PATHS ${legate_cython_rpaths})

# ########################################################################################
# * install targets-----------------------------------------------------------

include(CPack)

install(TARGETS legate_core_python DESTINATION ${lib_dir}
        EXPORT legate-core-python-exports)

if(_legate_core_FOUND_METHOD STREQUAL "PRE_BUILT")
  # If we found the library via pre-built libs then we also need to install them (and any
  # libs they depend on).
  string(JOIN
         "\n"
         code_string
         [=[
if(NOT DEFINED legate_core_DIR)
  set(legate_core_DIR ]=]
         "\"${legate_core_DIR}\""
         [=[)
endif()

execute_process(COMMAND
   "${CMAKE_COMMAND}"
    --install "${legate_core_DIR}"
    --prefix  "${CMAKE_INSTALL_PREFIX}"
    --verbose
  COMMAND_ECHO           STDOUT
  COMMAND_ERROR_IS_FATAL ANY
  ECHO_OUTPUT_VARIABLE
  ECHO_ERROR_VARIABLE
)
]=])
  install(CODE "${code_string}")
  # NOTE: currently not used since we forcibly install everything, but perhaps this might
  # be useful down the line.
  #
  # If we found the library via pre-built libs then we also need to install them (and any
  # libs they depend on). include(cmake/Modules/wheel_helpers.cmake)

  # install_imported_rt_deps( TARGET              legate::core RT_DEPS
  # legate-core-python-rt-deps LIBRARY_DESTINATION ${lib_dir} RUNTIME_DESTINATION
  # ${CMAKE_INSTALL_BINDIR} )
endif()

# ########################################################################################
# * install export -----------------------------------------------------------

set(doc_string
    [=[
Provide targets for Legate Python, the Foundation for All Legate Libraries.

Imported Targets:
  - legate::core_python

]=])

set(code_string "")

rapids_export(INSTALL legate_core_python
              EXPORT_SET legate-core-python-exports
              GLOBAL_TARGETS core_python
              NAMESPACE legate::
              DOCUMENTATION doc_string
              FINAL_CODE_BLOCK code_string)

# build export targets
rapids_export(BUILD legate_core_python
              EXPORT_SET legate-core-python-exports
              GLOBAL_TARGETS core_python
              NAMESPACE legate::
              DOCUMENTATION doc_string
              FINAL_CODE_BLOCK code_string)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
