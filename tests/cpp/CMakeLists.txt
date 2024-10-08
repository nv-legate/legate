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

cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)

project(legate_tests VERSION 0.1 LANGUAGES C CXX)

if(PROJECT_IS_TOP_LEVEL)
  # To catch people trying to build the tests from within tests/cpp instead of top-level
  message(FATAL_ERROR "Error: Tests can only be built as part of the main library build. Please re-run cmake from top-level directory (\${LEGATE_DIR}) with -Dlegate_BUILD_TESTS=ON"
  )
endif()

if(Legion_USE_CUDA)
  enable_language(CUDA)
endif()

include(rapids-test)

rapids_test_init()

include(${rapids-cmake-dir}/cpm/gtest.cmake)

# BUILD_EXPORT_SET and INSTALL_EXPORT_SET are crucial, otherwise gtest does not get
# installed
rapids_cpm_gtest(BUILD_EXPORT_SET legate-exports INSTALL_EXPORT_SET legate-exports
                                                                    CPM_ARGS SYSTEM TRUE)

include("${LEGATE_CMAKE_DIR}/Modules/debug_symbols.cmake")
include("${LEGATE_CMAKE_DIR}/Modules/clang_tidy.cmake")

set(unique_src "")

function(legate_configure_test)
  set(options)
  set(one_value NAME GPUS PERCENT)
  set(multi_value SOURCES)
  cmake_parse_arguments(_LEGATE_TEST "${options}" "${one_value}" "${multi_value}" ${ARGN})

  if(NOT _LEGATE_TEST_NAME)
    message(FATAL_ERROR "Must pass NAME")
  endif()

  if(NOT DEFINED _LEGATE_TEST_GPUS AND NOT DEFINED _LEGATE_TEST_PERCENT)
    set(_LEGATE_TEST_GPUS 0)
    set(_LEGATE_TEST_PERCENT 1)
  elseif(NOT DEFINED _LEGATE_TEST_GPUS)
    set(_LEGATE_TEST_GPUS 0)
  elseif(NOT DEFINED _LEGATE_TEST_PERCENT)
    set(_LEGATE_TEST_PERCENT 1)
  endif()

  add_executable(${_LEGATE_TEST_NAME} ${_LEGATE_TEST_SOURCES})
  set_target_properties(${_LEGATE_TEST_NAME}
                        PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                   "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}>"
                                   BUILD_RPATH_USE_ORIGIN TRUE
                                   # Don't use CMAKE_INSTALL_LIBDIR since we want the
                                   # rpath to be relative
                                   INSTALL_RPATH "${LEGATE_PLATFORM_RPATH_ORIGIN}/../lib"
                                   INSTALL_RPATH_USE_LINK_PATH TRUE
                                   CXX_STANDARD ${CMAKE_CXX_STANDARD}
                                   CXX_STANDARD_REQUIRED ON
                                   # For std:: support of __int128_t. Can be removed once
                                   # using cuda::std
                                   CXX_EXTENSIONS ON)

  target_include_directories(${_LEGATE_TEST_NAME} PRIVATE ${PROJECT_SOURCE_DIR})

  target_link_libraries(${_LEGATE_TEST_NAME}
                        PRIVATE legate::legate
                                GTest::gtest
                                GTest::gmock
                                # Some test modules require delving into the private
                                # headers of legate, which may contain fmt headers. As a
                                # result, we also need to add the fmt include paths etc.
                                # This is not done automatically for us, since fmt is a
                                # PRIVATE dependency of legate (as it is not found in any
                                # public includes).
                                fmt::fmt)

  if(Legion_USE_CUDA)
    set_target_properties(${_LEGATE_TEST_NAME}
                          PROPERTIES CUDA_ARCHITECTURES ${Legion_CUDA_ARCH}
                                     CUDA_STANDARD ${CMAKE_CUDA_STANDARD}
                                     CUDA_STANDARD_REQUIRED ON)
    target_link_libraries(${_LEGATE_TEST_NAME} PRIVATE NCCL::NCCL)
  endif()

  legate_add_target_compile_option(${_LEGATE_TEST_NAME} CXX PRIVATE legate_CXX_OPTIONS)
  legate_add_target_compile_option(${_LEGATE_TEST_NAME} CUDA PRIVATE legate_CUDA_OPTIONS)

  legate_add_target_compile_option(${_LEGATE_TEST_NAME} CXX PRIVATE legate_CXX_FLAGS)
  legate_add_target_compile_option(${_LEGATE_TEST_NAME} CUDA PRIVATE legate_CUDA_FLAGS)

  legate_add_target_link_option(${_LEGATE_TEST_NAME} PRIVATE legate_LINKER_FLAGS)

  rapids_test_add(NAME ${_LEGATE_TEST_NAME}
                  COMMAND ${_LEGATE_TEST_NAME}
                  GPUS ${_LEGATE_TEST_GPUS}
                  PERCENT ${_LEGATE_TEST_PERCENT}
                  INSTALL_COMPONENT_SET testing)

  # TODO(jfaibussowit): We need to use DISCOVERY_MODE PRE_TEST (as described below), but
  # CMake has a bug in their implementation. See
  # https://gitlab.kitware.com/cmake/cmake/-/issues/26294
  #
  # cmake-format: off
  # gtest_discover_tests(${_LEGATE_TEST_NAME}
  #                      PROPERTIES ENVIRONMENT
  #                                 LEGATE_TEST=1
  #                                 # Cannot discover tests at build time because CI splits
  #                                 # build and run into 2 separate phase. The build phase
  #                                 # does not necessarily have the required libs to run the
  #                                 # execute.
  #                                 DISCOVERY_MODE
  #                                 PRE_TEST)
  # cmake-format: on

  legate_install_debug_symbols(TARGET ${_LEGATE_TEST_NAME}
                               INSTALL_DIR ${CMAKE_INSTALL_BINDIR})

  list(APPEND LEGATE_CONFIGURED_TESTS ${_LEGATE_TEST_NAME})
  # propagate the updated list upwards
  set(LEGATE_CONFIGURED_TESTS ${LEGATE_CONFIGURED_TESTS} PARENT_SCOPE)

  foreach(src IN LISTS _LEGATE_TEST_SOURCES)
    # Because each of the tests share a common main src file, we need to make sure we
    # don't add a tidy target for it twice
    list(FIND unique_src "${src}" idx)
    if(idx EQUAL -1)
      # not found
      legate_add_tidy_target(SOURCE "${src}")
      list(APPEND unique_src "${src}")
    endif()
  endforeach()
  set(unique_src "${unique_src}" PARENT_SCOPE)
endfunction()

if(Legion_USE_CUDA)
  file(GLOB integration_GPU_SRC ${PROJECT_SOURCE_DIR}/integration/*.cu)
endif()

file(GLOB main_with_runtime_init_SRC ${PROJECT_SOURCE_DIR}/main_with_runtime_init.cc)
file(GLOB main_wo_runtime_init_SRC ${PROJECT_SOURCE_DIR}/main_wo_runtime_init.cc)

file(GLOB bug_SRC ${PROJECT_SOURCE_DIR}/bug/*.cc)
file(GLOB integration_SRC ${PROJECT_SOURCE_DIR}/integration/*.cc)
file(GLOB tasks_SRC ${PROJECT_SOURCE_DIR}/integration/tasks/*.cc)
file(GLOB unit_SRC ${PROJECT_SOURCE_DIR}/unit/*.cc)
file(GLOB stl_SRC ${PROJECT_SOURCE_DIR}/experimental/stl/*.cc)
file(GLOB experimental_SRC ${PROJECT_SOURCE_DIR}/experimental/*.cc)

file(GLOB noinit_SRC ${PROJECT_SOURCE_DIR}/noinit/*.cc)
file(GLOB non_reentrant_SRC ${PROJECT_SOURCE_DIR}/non_reentrant/*.cc)

include(GNUInstallDirs)

# See https://github.com/rapidsai/rapids-cmake/issues/583
if("${rapids-cmake-version}" VERSION_LESS "24.06.00")
  string(JOIN
         "\n"
         code_string
         [[
if(POLICY CMP0159)
  cmake_policy(SET CMP0159 OLD)
  set(CMAKE_POLICY_DEFAULT_CMP0159 OLD)
endif()
]])
  install(CODE "${code_string}")
endif()

legate_configure_test(NAME tests_with_runtime
                      SOURCES ${main_with_runtime_init_SRC} ${bug_SRC} ${tasks_SRC}
                              ${integration_SRC} ${unit_SRC} ${experimental_SRC}
                              ${stl_SRC})
legate_configure_test(NAME tests_wo_runtime SOURCES ${main_wo_runtime_init_SRC}
                                                    ${noinit_SRC})
legate_configure_test(NAME tests_non_reentrant SOURCES ${main_with_runtime_init_SRC}
                                                       ${non_reentrant_SRC})

if(integration_GPU_SRC)
  legate_configure_test(NAME tests_with_gpus
                        SOURCES ${main_with_runtime_init_SRC} ${bug_SRC} ${tasks_SRC}
                                ${integration_SRC} ${unit_SRC} ${experimental_SRC}
                                ${stl_SRC}
                        GPUS 1
                        PERCENT 30)
endif()

rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing
                                DESTINATION ${CMAKE_INSTALL_BINDIR} INCLUDE_IN_ALL)

function(legate_create_dummy_cuda_driver)
  add_library(legate_dummy_cuda_driver utilities/dummy_cuda_driver.cc)
  set_target_properties(legate_dummy_cuda_driver PROPERTIES POSITION_INDEPENDENT_CODE ON
                                                            LIBRARY_OUTPUT_DIRECTORY lib)
  # tests_wo_runtime contains the CUDADriverAPI test which tests whether we can
  # successfully load a shim CUDA driver module. By adding the RPATH, we can just pass
  # "liblegate_dummy_cuda_driver.so" to the loader, and because its in the RPATH, dlopen()
  # will find it.
  #
  # This works because dir structure is like so:
  #
  # cmake-format: off
  # CMAKE_CURRENT_BINARY_DIR/
  # | - bin/
  # |   | - tests_with_runtime
  # | - lib/
  #     | - liblegate_dummy_cuda_driver.dylib
  # cmake-format: on
  #
  # As an added bonus, CMake will also ensure this rpath gets updated when both of these
  # get installed.
  set_property(TARGET tests_wo_runtime APPEND
               PROPERTY BUILD_RPATH "${LEGATE_PLATFORM_RPATH_ORIGIN}/../lib")
  install(TARGETS legate_dummy_cuda_driver DESTINATION ${lib_dir})
endfunction()
legate_create_dummy_cuda_driver()
