#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

##############################################################################
# - conda environment --------------------------------------------------------

include(GNUInstallDirs)

rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)

# We're building python extension libraries, which must always be installed
# under lib/, even if the system normally uses lib64/. Rapids-cmake currently
# doesn't realize this when we're going through scikit-build, see
# https://github.com/rapidsai/rapids-cmake/issues/426
if(TARGET conda_env)
  set(CMAKE_INSTALL_LIBDIR "lib")
endif()

rapids_cmake_install_lib_dir(lib_dir)

if(legate_core_SETUP_PY_MODE STREQUAL "develop")
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
endif()

##############################################################################
# - Find or build the C++ core -----------------------------------------------

include(cmake/Modules/find_legate_core_cpp.cmake)

find_legate_core_cpp(
  VERSION            ${legate_core_version}
  BUILD_EXPORT_SET   legate-core-python-exports
  INSTALL_EXPORT_SET legate-core-python-exports
)

##############################################################################
# - Cython build -------------------------------------------------------------

rapids_find_package(Python3
  GLOBAL_TARGETS     Python3::Module
  BUILD_EXPORT_SET   legate-core-python-exports
  INSTALL_EXPORT_SET legate-core-python-exports
  FIND_ARGS
    REQUIRED
    COMPONENTS Development
)

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

if(legate_core_SETUP_PY_MODE STREQUAL "develop")
  # If we are doing an editable install, then the cython rpaths need to point back to the
  # original (uninstalled) legate libs, since otherwise it cannot find them.
  #
  # Use of legate_core vs legate::core is deliberate! The former may or may not be defined
  # depending on whether we built legate.core while legate::core is always defined if
  # imported.
  if(TARGET legate_core)
    get_target_property(cython_lib_dir legate_core LIBRARY_OUTPUT_DIRECTORY)
    get_target_property(legate_cpp_dir legate_core BINARY_DIR)
    if(legate_cpp_dir)
      set(cython_lib_dir "${legate_cpp_dir}/${cython_lib_dir}")
    endif()
  else()
    get_target_property(cython_lib_dir legate::core LOCATION)
    cmake_path(GET cython_lib_dir PARENT_PATH cython_lib_dir)
  endif()
else()
  # This somehow sets the rpath correctly for regular
  # installs. rapids_cython_add_rpath_entries() mentions that:
  #
  # PATHS may either be absolute or relative to the ROOT_DIRECTORY. The paths are always
  # converted to be relative to the current directory i.e relative to $ORIGIN in the
  # RPATH.
  #
  # where
  #
  # ROOT_DIRECTORY "Defaults to ${PROJECT_SOURCE_DIR}".
  #
  # Since there is nothing interesting 2 directories up from PROJECT_SOURCE_DIR, my best
  # guess is that the 2 directories up refers to 2 directories up from the python
  # site-packages dir, which is always found as
  # /path/to/lib/python3.VERSION/site-packages/. The combined rpaths would make this point
  # to /path/to/lib which seems right. But who knows.
  set(cython_lib_dir "../../")
endif()

message(STATUS "legate_core_python: cython_lib_dir='${cython_lib_dir}'")

rapids_cython_add_rpath_entries(TARGET legate::core PATHS "${cython_lib_dir}")

# Legion sets this to "OFF" if not enabled, normalize it to an empty list instead
if(NOT Legion_NETWORKS)
  set(Legion_NETWORKS "")
endif()

# NOTE: if you need any of these values to be guaranteed to be defined, add them to
# legate_core_SUBDIR_CMAKE_EXPORT_VARS above!
add_custom_target(generate_install_info_py ALL
  COMMAND ${CMAKE_COMMAND}
  -DLegion_NETWORKS="${Legion_NETWORKS}"
  -DGASNet_CONDUIT="${GASNet_CONDUIT}"
  -DLegion_USE_CUDA="${Legion_USE_CUDA}"
  -DLegion_USE_OpenMP="${Legion_USE_OpenMP}"
  -DLegion_MAX_DIM="${Legion_MAX_DIM}"
  -DLegion_MAX_FIELDS="${Legion_MAX_FIELDS}"
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  -DLEGATE_CORE_DIR="${LEGATE_CORE_DIR}"
  -DLEGATE_CORE_ARCH="${LEGATE_CORE_ARCH}"
  -Dlegate_core_LIB_NAME="$<TARGET_FILE_PREFIX:legate::core>$<TARGET_FILE_BASE_NAME:legate::core>"
  -Dlegate_core_FULL_LIB_NAME="$<TARGET_FILE_NAME:legate::core>"
  -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate_install_info_py.cmake"
  BYPRODUCTS ${CMAKE_CURRENT_SOURCE_DIR}/legate/install_info.py
  COMMENT "Generate install_info.py"
)

##############################################################################
# - install targets-----------------------------------------------------------

include(CPack)

install(TARGETS legate_core_python
  DESTINATION ${lib_dir}
  EXPORT      legate-core-python-exports
)

if(_legate_core_FOUND_METHOD STREQUAL "PRE_BUILT")
  # If we found the library via pre-built libs then we also need to install them (and any
  # libs they depend on).
  string(JOIN "\n" code_string
    [=[
if(NOT DEFINED legate_core_DIR)
  set(legate_core_DIR ]=] "\"${legate_core_DIR}\"" [=[)
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
]=]
  )
  install(CODE "${code_string}")
  # NOTE: currently not used since we forcibly install everything, but perhaps this might
  # be useful down the line.
  #
  # If we found the library via pre-built libs then we also need to install them (and any
  # libs they depend on).
  # include(cmake/Modules/wheel_helpers.cmake)

  # install_imported_rt_deps(
  #   TARGET              legate::core
  #   RT_DEPS             legate-core-python-rt-deps
  #   LIBRARY_DESTINATION ${lib_dir}
  #   RUNTIME_DESTINATION ${CMAKE_INSTALL_BINDIR}
  # )
endif()

##############################################################################
# - install export -----------------------------------------------------------

set(doc_string
  [=[
Provide targets for Legate Python, the Foundation for All Legate Libraries.

Imported Targets:
  - legate::core_python

]=])

set(code_string "")

rapids_export(
  INSTALL legate_core_python
  EXPORT_SET legate-core-python-exports
  GLOBAL_TARGETS core_python
  NAMESPACE legate::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string)

# build export targets
rapids_export(
  BUILD legate_core_python
  EXPORT_SET legate-core-python-exports
  GLOBAL_TARGETS core_python
  NAMESPACE legate::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
