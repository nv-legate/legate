#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#=============================================================================

list(APPEND CMAKE_MESSAGE_CONTEXT "python")

set(LEGATE_PYTHON_PKG_ROOT "${CMAKE_CURRENT_LIST_DIR}")

if(SKBUILD_STATE STREQUAL "editable")
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
endif()

# ########################################################################################
# * Find or build the C++ -----------------------------------------------

include("${LEGATE_CMAKE_DIR}/Modules/find_legate_cpp.cmake")

find_legate_cpp(VERSION ${PROJECT_VERSION} BUILD_EXPORT_SET legate-python-exports
                INSTALL_EXPORT_SET legate-python-exports)

# ########################################################################################
# * Cython build -------------------------------------------------------------

rapids_find_package(Python3
                    GLOBAL_TARGETS Python3::Module
                    BUILD_EXPORT_SET legate-python-exports
                    INSTALL_EXPORT_SET legate-python-exports
                    FIND_ARGS
                    REQUIRED
                    COMPONENTS Development)

include(rapids-cython-core)

rapids_cython_init()

add_library(legate_python INTERFACE)
add_library(legate::legate_python ALIAS legate_python)
target_link_libraries(legate_python INTERFACE legate::legate)

add_subdirectory(legate)

include("${LEGATE_CMAKE_DIR}/Modules/cython_rpaths.cmake")

legate_populate_cython_dependency_rpaths(RAPIDS_ASSOCIATED_TARGET legate::legate
                                         ROOT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

# ########################################################################################
# * install targets-----------------------------------------------------------

include(CPack)

install(TARGETS legate_python DESTINATION ${CMAKE_INSTALL_LIBDIR}
        EXPORT legate-python-exports)

if(_legate_FOUND_METHOD STREQUAL "PRE_BUILT")
  # If we found the library via pre-built libs then we also need to install them (and any
  # libs they depend on).
  string(JOIN
         "\n"
         code_string
         [=[
if(NOT DEFINED legate_DIR)
  set(legate_DIR ]=]
         "\"${legate_DIR}\""
         [=[)
endif()

execute_process(COMMAND
   "${CMAKE_COMMAND}"
    --install "${legate_DIR}"
    --prefix  "${CMAKE_INSTALL_PREFIX}"
    --verbose
  COMMAND_ECHO           STDOUT
  COMMAND_ERROR_IS_FATAL ANY
  ECHO_OUTPUT_VARIABLE
  ECHO_ERROR_VARIABLE
)
]=])
  install(CODE "${code_string}")
endif()

# ########################################################################################
# * install export -----------------------------------------------------------

set(doc_string
    [=[
Provide targets for Legate Python, the Foundation for All Legate Libraries.

Imported Targets:
  - legate::legate_python

]=])

set(code_string "")

rapids_export(INSTALL legate_python
              EXPORT_SET legate-python-exports
              GLOBAL_TARGETS legate_python
              NAMESPACE legate::
              DOCUMENTATION doc_string
              FINAL_CODE_BLOCK code_string)

# build export targets
rapids_export(BUILD legate_python
              EXPORT_SET legate-python-exports
              GLOBAL_TARGETS legate_python
              NAMESPACE legate::
              DOCUMENTATION doc_string
              FINAL_CODE_BLOCK code_string)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
