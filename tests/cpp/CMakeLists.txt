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

cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)

cmake_path(SET LEGATE_CORE_DIR NORMALIZE "${CMAKE_CURRENT_SOURCE_DIR}/../../")

include(${LEGATE_CORE_DIR}/cmake/policies.cmake NO_POLICY_SCOPE)

project(cpp_tests VERSION 0.1 LANGUAGES C CXX)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:$ORIGIN")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if (NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 17)
endif()
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CUDA_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os")
set(CMAKE_CUDA_FLAGS_MINSIZEREL "-Os")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O2 -g")

# Needed to integrate with LLVM/clang tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${LEGATE_CORE_DIR}/cmake/Modules/ensure_legate.cmake)

legate_ensure_legate()

include(CTest)

include(${LEGATE_CORE_DIR}/cmake/Modules/include_rapids.cmake)

legate_include_rapids()

rapids_cpm_init()

if(Legion_USE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  enable_language(CUDA)

  file(GLOB integration_GPU_SRC ${PROJECT_SOURCE_DIR}/integration/*.cu)

  include(cmake/thirdparty/get_nccl.cmake)
else()
  # rapids_test_init() calls find_package(CUDAToolKit REQUIRED) if it does not find a
  # generate_resource_spec executable at exactly at
  # <build-dir>/rapids-cmake/generate_resource_spec
  #
  # Obviously this doesn't work so well on cpu-only builds, so we spoof the file
  # ourselves. We first find the source file, patch out any CUDA-related nonsense, and
  # then compile ourselves
  include(${rapids-cmake-dir}/test/detail/default_names.cmake)

  set(patch_file "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/generate_resource_spec.diff")
  set(eval_file ${rapids-cmake-dir}/test/detail/generate_resource_spec.cpp)
  set(eval_exe ${CMAKE_CURRENT_BINARY_DIR}/rapids-cmake/${rapids_test_generate_exe_name})

  if(LEGATE_TESTS_PATCHED_EVAL_FILE VERSION_EQUAL rapids-cmake-version)
    message("-- Spec generator already patched for rapids version ${rapids-cmake-version}, no need to re-patch")
  else()
    message("-- Patching ${eval_file}")
    find_program(PATCH NAMES patch REQUIRED)
    execute_process(
      COMMAND ${PATCH} --unified ${eval_file} --input=${patch_file} --ignore-whitespace --quiet
      TIMEOUT 15
      COMMAND_ERROR_IS_FATAL ANY
    )
    set(LEGATE_TESTS_PATCHED_EVAL_FILE "${rapids-cmake-version}" CACHE INTERNAL "" FORCE)
  endif()

  # Compile the (now patched) resource spec generator, and place the result where rapids
  # expects to find it. try_compile() is a bit of a misnomer here -- this compilation MUST
  # succeed, as rapids_test_init() below will fail if it doesn't
  try_compile(
    SUCCEEDED
    SOURCES ${eval_file}
    COPY_FILE ${eval_exe}
    OUTPUT_VARIABLE compile_ret
  )

  if(NOT SUCCEEDED)
    message(FATAL_ERROR "Failed to compile patched resource spec generator for rapids_cmake_test(): ${compile_ret}" )
  endif()
endif()

include(rapids-test)

rapids_test_init()

include(${rapids-cmake-dir}/cpm/gtest.cmake)

rapids_cpm_gtest()

include(${LEGATE_CORE_DIR}/cmake/Modules/debug_symbols.cmake)

function(ConfigureTest CMAKE_TEST_NAME)
  set(options)
  set(one_value GPUS PERCENT)
  set(multi_value)
  cmake_parse_arguments(_LEGATE_TEST "${options}" "${one_value}" "${multi_value}" ${ARGN})
  if(NOT DEFINED _LEGATE_TEST_GPUS AND NOT DEFINED _LEGATE_TEST_PERCENT)
    set(_LEGATE_TEST_GPUS 0)
    set(_LEGATE_TEST_PERCENT 1)
  elseif(NOT DEFINED _LEGATE_TEST_GPUS)
    set(_LEGATE_TEST_GPUS 0)
  elseif(NOT DEFINED _LEGATE_TEST_PERCENT)
    set(_LEGATE_TEST_PERCENT 1)
  endif()

  add_executable(${CMAKE_TEST_NAME} ${_LEGATE_TEST_UNPARSED_ARGUMENTS})
  set_target_properties(
    ${CMAKE_TEST_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}>"
               INSTALL_RPATH "\$ORIGIN/../../../lib"
               CXX_STANDARD ${CMAKE_CXX_STANDARD}
               CXX_STANDARD_REQUIRED ON
               # For std:: support of __int128_t. Can be removed once using cuda::std
               CXX_EXTENSIONS ON
               CUDA_STANDARD ${CMAKE_CUDA_STANDARD}
               CUDA_STANDARD_REQUIRED ON
  )

  target_include_directories(${CMAKE_TEST_NAME} PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(${CMAKE_TEST_NAME} PRIVATE legate::core PRIVATE GTest::gtest)

  if(Legion_USE_CUDA)
    target_link_libraries(${CMAKE_TEST_NAME} PRIVATE NCCL::NCCL)
  endif()

  set_property(TARGET ${CMAKE_TEST_NAME} PROPERTY ENABLE_EXPORTS ON)

  rapids_test_add(
    NAME ${CMAKE_TEST_NAME}
    COMMAND ${CMAKE_TEST_NAME}
    GPUS ${_LEGATE_TEST_GPUS}
    PERCENT ${_LEGATE_TEST_PERCENT}
    INSTALL_COMPONENT_SET testing
  )

  legate_core_debug_syms(${CMAKE_TEST_NAME} INSTALL_DIR ${CMAKE_INSTALL_BINDIR})
endfunction()

file(GLOB main_SRC ${PROJECT_SOURCE_DIR}/main.cc)
file(GLOB bug_SRC ${PROJECT_SOURCE_DIR}/bug/*.cc)
file(GLOB integration_SRC ${PROJECT_SOURCE_DIR}/integration/*.cc)
file(GLOB tasks_SRC ${PROJECT_SOURCE_DIR}/integration/tasks/*.cc)
file(GLOB unit_SRC ${PROJECT_SOURCE_DIR}/unit/*.cc)

include(GNUInstallDirs)

ConfigureTest(
  cpp_tests
  ${main_SRC} ${bug_SRC} ${tasks_SRC} ${integration_SRC} ${unit_SRC}
)

if(integration_GPU_SRC)
  ConfigureTest(
    gpu_tests
    ${main_SRC} ${integration_GPU_SRC}
    GPUS 1
    PERCENT 30
  )
endif()

# so CI has an easier time of picking up the installers
if (NOT CPACK_GENERATOR)
  set(CPACK_GENERATOR STGZ)
endif()
if (NOT CPACK_PACKAGE_FILE_NAME)
  set(CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}")
endif()

include(CPack)

rapids_test_install_relocatable(
  INSTALL_COMPONENT_SET testing
  DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDE_IN_ALL
)
