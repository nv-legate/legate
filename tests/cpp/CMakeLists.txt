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

project(legate_core_tests VERSION 0.1 LANGUAGES C CXX)

if(PROJECT_IS_TOP_LEVEL)
  # To catch people trying to build the tests from within tests/cpp instead of top-level
  message(FATAL_ERROR "Error: Tests can only be built as part of the main library build. Please re-run cmake from top-level directory (\${LEGATE_CORE_DIR}) with -Dlegate_core_BUILD_TESTS=ON")
endif()

if(Legion_USE_CUDA)
  enable_language(CUDA)
endif()

include(rapids-test)

rapids_test_init()

include(${rapids-cmake-dir}/cpm/gtest.cmake)

# BUILD_EXPORT_SET and INSTALL_EXPORT_SET are crucial, otherwise gtest does not get
# installed
rapids_cpm_gtest(
  BUILD_EXPORT_SET   legate-core-exports
  INSTALL_EXPORT_SET legate-core-exports
)

include(${LEGATE_CORE_DIR}/cmake/Modules/debug_symbols.cmake)

function(ConfigureTest CMAKE_TEST_NAME)
  set(options)
  set(one_value GPUS PERCENT)
  set(multi_value SOURCES)
  cmake_parse_arguments(_LEGATE_TEST "${options}" "${one_value}" "${multi_value}" ${ARGN})
  if(NOT DEFINED _LEGATE_TEST_GPUS AND NOT DEFINED _LEGATE_TEST_PERCENT)
    set(_LEGATE_TEST_GPUS 0)
    set(_LEGATE_TEST_PERCENT 1)
  elseif(NOT DEFINED _LEGATE_TEST_GPUS)
    set(_LEGATE_TEST_GPUS 0)
  elseif(NOT DEFINED _LEGATE_TEST_PERCENT)
    set(_LEGATE_TEST_PERCENT 1)
  endif()

  add_executable(${CMAKE_TEST_NAME} ${_LEGATE_TEST_SOURCES})
  set_target_properties(
    ${CMAKE_TEST_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}>"
               BUILD_RPATH_USE_ORIGIN      TRUE
               # Don't use CMAKE_INSTALL_LIBDIR since we want the rpath to be relative
               INSTALL_RPATH               "${LEGATE_CORE_PLATFORM_RPATH_ORIGIN}/../lib"
               INSTALL_RPATH_USE_LINK_PATH TRUE
               CXX_STANDARD                ${CMAKE_CXX_STANDARD}
               CXX_STANDARD_REQUIRED       ON
               # For std:: support of __int128_t. Can be removed once using cuda::std
               CXX_EXTENSIONS              ON
  )

  target_include_directories(
    ${CMAKE_TEST_NAME}
    PRIVATE
    ${PROJECT_SOURCE_DIR})

  target_link_libraries(${CMAKE_TEST_NAME}
    PRIVATE
      legate::core
      GTest::gtest
  )

  if(Legion_USE_CUDA)
    set_target_properties(
      ${CMAKE_TEST_NAME}
      PROPERTIES
        CUDA_ARCHITECTURES     ${Legion_CUDA_ARCH}
        CUDA_STANDARD          ${CMAKE_CUDA_STANDARD}
        CUDA_STANDARD_REQUIRED ON
    )
    target_link_libraries(${CMAKE_TEST_NAME} PRIVATE NCCL::NCCL)
  endif()

  legate_core_add_target_compile_option(${CMAKE_TEST_NAME} CXX PRIVATE legate_core_CXX_OPTIONS)
  legate_core_add_target_compile_option(${CMAKE_TEST_NAME} CUDA PRIVATE legate_core_CUDA_OPTIONS)

  legate_core_add_target_compile_option(${CMAKE_TEST_NAME} CXX PRIVATE legate_core_CXX_FLAGS)
  legate_core_add_target_compile_option(${CMAKE_TEST_NAME} CUDA PRIVATE legate_core_CUDA_FLAGS)

  legate_core_add_target_link_option(${CMAKE_TEST_NAME} PRIVATE legate_core_LINKER_FLAGS)

  rapids_test_add(
    NAME ${CMAKE_TEST_NAME}
    COMMAND ${CMAKE_TEST_NAME}
    GPUS ${_LEGATE_TEST_GPUS}
    PERCENT ${_LEGATE_TEST_PERCENT}
    INSTALL_COMPONENT_SET testing
  )

  legate_core_debug_syms(${CMAKE_TEST_NAME} INSTALL_DIR ${CMAKE_INSTALL_BINDIR})

  list(APPEND LEGATE_CORE_CONFIGURED_TESTS ${CMAKE_TEST_NAME})
  # propagate the updated list upwards
  set(LEGATE_CORE_CONFIGURED_TESTS ${LEGATE_CORE_CONFIGURED_TESTS} PARENT_SCOPE)

  list(APPEND legate_core_TIDY_SOURCES ${_LEGATE_TEST_SOURCES})
  set(legate_core_TIDY_SOURCES ${legate_core_TIDY_SOURCES} PARENT_SCOPE)
endfunction()

if(Legion_USE_CUDA)
  file(GLOB integration_GPU_SRC ${PROJECT_SOURCE_DIR}/integration/*.cu)
endif()

file(GLOB main_SRC ${PROJECT_SOURCE_DIR}/main.cc)
file(GLOB bug_SRC ${PROJECT_SOURCE_DIR}/bug/*.cc)
file(GLOB integration_SRC ${PROJECT_SOURCE_DIR}/integration/*.cc)
file(GLOB tasks_SRC ${PROJECT_SOURCE_DIR}/integration/tasks/*.cc)
file(GLOB unit_SRC ${PROJECT_SOURCE_DIR}/unit/*.cc)
file(GLOB stl_SRC ${PROJECT_SOURCE_DIR}/stl/*.cc)
file(GLOB experimental_SRC ${PROJECT_SOURCE_DIR}/experimental/*.cc)

include(GNUInstallDirs)

ConfigureTest(
  cpp_tests
  SOURCES ${main_SRC} ${bug_SRC} ${tasks_SRC} ${integration_SRC} ${unit_SRC} ${experimental_SRC} ${stl_SRC}
)

if(integration_GPU_SRC)
  ConfigureTest(
    gpu_tests
    SOURCES ${main_SRC} ${tasks_SRC} ${integration_SRC} ${integration_GPU_SRC}
    GPUS 1
    PERCENT 30
  )
endif()

rapids_test_install_relocatable(
  INSTALL_COMPONENT_SET testing
  DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDE_IN_ALL
)

# Add a dummy target whose only job it is to depend on the tests being built.
add_custom_target(build_tests)
add_dependencies(build_tests ${LEGATE_CORE_CONFIGURED_TESTS})

# propagate to main CMakeLists.txt
set(legate_core_TIDY_SOURCES ${legate_core_TIDY_SOURCES} PARENT_SCOPE)
