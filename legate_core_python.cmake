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

##############################################################################
# - User Options  ------------------------------------------------------------

option(FIND_LEGATE_CORE_CPP "Search for existing legate_core C++ installations before defaulting to local files"
       OFF)

##############################################################################
# - Dependencies -------------------------------------------------------------

# If the user requested it we attempt to find legate_core.
if(FIND_LEGATE_CORE_CPP)
  include("${rapids-cmake-dir}/export/detail/parse_version.cmake")
  rapids_export_parse_version(${legate_core_version} legate_core parsed_ver)
  rapids_find_package(legate_core ${parsed_ver} EXACT CONFIG
                      GLOBAL_TARGETS     legate::core
                      BUILD_EXPORT_SET   legate-core-python-exports
                      INSTALL_EXPORT_SET legate-core-python-exports)
else()
  set(legate_core_FOUND OFF)
endif()

if(NOT legate_core_FOUND)
  set(SKBUILD OFF)
  set(Legion_USE_Python ON)
  add_subdirectory(. legate-core-cpp)
  set(SKBUILD ON)
endif()

add_custom_target("generate_install_info_py" ALL
  COMMAND ${CMAKE_COMMAND}
          -DLegion_NETWORKS="${Legion_NETWORKS}"
          -DGASNet_CONDUIT="${GASNet_CONDUIT}"
          -DLegion_USE_CUDA="${Legion_USE_CUDA}"
          -DLegion_USE_OpenMP="${Legion_USE_OpenMP}"
          -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
          -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
          -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate_install_info_py.cmake"
  COMMENT "Generate install_info.py"
)

add_library(legate_core_python INTERFACE)
add_library(legate::core_python ALIAS legate_core_python)
target_link_libraries(legate_core_python INTERFACE legate::core)

include(rapids-cython)
rapids_cython_init()

add_subdirectory(legate/core/_lib)
add_subdirectory(legate/timing/_lib)

set(cython_lib_dir "../../")

if(CMAKE_INSTALL_RPATH_USE_LINK_PATH)
  if(NOT TARGET legate_core)
    get_target_property(cython_lib_dir legate::core LOCATION)
    cmake_path(GET cython_lib_dir PARENT_PATH cython_lib_dir)
  else()
    get_target_property(cython_lib_dir legate_core LIBRARY_OUTPUT_DIRECTORY)
    get_target_property(legate_cpp_dir legate_core BINARY_DIR)
    if(legate_cpp_dir)
      set(cython_lib_dir "${legate_cpp_dir}/${cython_lib_dir}")
    endif()
  endif()
endif()

message(STATUS "legate_core_python: cython_lib_dir='${cython_lib_dir}'")

rapids_cython_add_rpath_entries(TARGET legate_core PATHS "${cython_lib_dir}")

##############################################################################
# - conda environment --------------------------------------------------------

rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)

# We're building python extension libraries, which must always be installed
# under lib/, even if the system normally uses lib64/. Rapids-cmake currently
# doesn't realize this when we're going through scikit-build, see
# https://github.com/rapidsai/rapids-cmake/issues/426
if(TARGET conda_env)
  set(CMAKE_INSTALL_LIBDIR "lib")
endif()

##############################################################################
# - install targets-----------------------------------------------------------

include(CPack)
include(GNUInstallDirs)
rapids_cmake_install_lib_dir(lib_dir)

install(TARGETS legate_core_python
        DESTINATION ${lib_dir}
        EXPORT legate-core-python-exports)

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

# We want to extract where the _skbuild/<machine-specific>/cmake-build folder is when
# using scikit-build to build the project. scikit-build expects all build artifacts to go
# there after configuration, but this may not always be the case. The user may pass -B
# ./path/to/other/build, or this may be set automatically by a cmake preset.
#
# The workaround is therefore to locate this cmake-build folder, and -- after
# configuration is complete -- create a symlink from the *actual* build folder to
# cmake-build.
#
# But locating cmake-build is not easy:
#
# - CMAKE_CURRENT_BINARY_DIR is unreliable since it may be overridden (as detailed above).
# - CMAKE_CURRENT_[SOURCE|LIST]_DIR don't work, they are /path/to/legate.core.internal
#   (not the directory from which cmake was invoked)
#
# So the trick is to exploit the fact that scikit-build sets CMAKE_INSTALL_PREFIX to
# _skbuild/<machine-specific>/cmake-install (and enforces that the user does not override
# this! see
# https://github.com/scikit-build/scikit-build/blob/main/skbuild/cmaker.py#L321). From
# this, we can reconstruct the cmake-build path.
if(SKBUILD)
  cmake_path(GET CMAKE_INSTALL_PREFIX PARENT_PATH skbuild_root_dir)
  cmake_path(APPEND skbuild_root_dir "cmake-build" OUTPUT_VARIABLE skbuild_cmake_build_dir)
  if (NOT (${skbuild_cmake_build_dir} STREQUAL ${CMAKE_CURRENT_BINARY_DIR}))
    # The binary dir has been overridden.
    file(REMOVE_RECURSE ${skbuild_cmake_build_dir})
    file(CREATE_LINK ${CMAKE_CURRENT_BINARY_DIR} ${skbuild_cmake_build_dir} SYMBOLIC)
  endif()
endif()
