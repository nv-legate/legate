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

include(rapids-test)

rapids_test_init()

legate_find_or_configure(PACKAGE google_test)

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

  if(NOT DEFINED _LEGATE_TEST_GPUS)
    set(_LEGATE_TEST_GPUS 0)
  endif()
  if(NOT DEFINED _LEGATE_TEST_PERCENT)
    set(_LEGATE_TEST_PERCENT 1)
  endif()

  add_executable(${_LEGATE_TEST_NAME} ${_LEGATE_TEST_SOURCES})
  set_target_properties(${_LEGATE_TEST_NAME}
                        PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                   "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}>"
                                   BUILD_RPATH_USE_ORIGIN TRUE
                                   # Don't use CMAKE_INSTALL_LIBDIR since we want the
                                   # rpath to be relative
                                   INSTALL_RPATH
                                   "${LEGATE_PLATFORM_RPATH_ORIGIN}/../${CMAKE_INSTALL_LIBDIR};${LEGATE_PLATFORM_RPATH_ORIGIN}/../${legate_DEP_INSTALL_LIBDIR}"
                                   INSTALL_RPATH_USE_LINK_PATH TRUE)

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
                                fmt::fmt-header-only)

  if(Legion_USE_CUDA)
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

set(with_runtime_SRC
    # bug
    bug/alias_via_promote.cc
    bug/aligned_unpack.cc
    bug/dangling_store_partition.cc
    bug/logical_store_transform.cc
    bug/nullable_arrays.cc
    bug/premature_free.cc
    bug/singleton_index_task.cc
    # integration
    integration/aligned_unbound_stores.cc
    integration/alignment_constraints.cc
    integration/attach.cc
    integration/auto_task_error.cc
    integration/bloat_constraints.cc
    integration/broadcast_constraints.cc
    integration/child_store.cc
    integration/consensus_match.cc
    integration/copy_failure.cc
    integration/copy_gather.cc
    integration/copy_gather_scatter.cc
    integration/copy_normal.cc
    integration/copy_scatter.cc
    integration/cpu_communicator.cc
    integration/delinearize.cc
    integration/exception.cc
    integration/field_reuse.cc
    integration/fill.cc
    integration/find_memory_kind.cc
    integration/image_constraints.cc
    integration/index_attach.cc
    integration/inline_map.cc
    integration/inout.cc
    integration/is_running_in_task.cc
    integration/machine_scope.cc
    integration/manual_task.cc
    integration/manual_task_proj.cc
    integration/mixed_dim.cc
    integration/multi_scalar_out.cc
    integration/partitioner.cc
    integration/projection.cc
    integration/provenance.cc
    integration/redundant.cc
    integration/region_manager.cc
    integration/register_variants.cc
    integration/replicated_write.cc
    integration/req_analyzer.cc
    integration/scalar_out.cc
    integration/scale_constraints.cc
    integration/task_misc.cc
    integration/tasks/task_simple.cc
    integration/test_task_store.cc
    integration/tree_reduce.cc
    integration/tree_reduce_unique.cc
    integration/tunable.cc
    integration/variant_options_precedence.cc
    integration/weighted.cc
    # unit
    unit/buffer.cc
    unit/constraint.cc
    unit/debug.cc
    unit/dispatch.cc
    unit/library.cc
    unit/machine.cc
    unit/mapping.cc
    unit/physical_store.cc
    unit/registration.cc
    unit/runtime.cc
    unit/scalar.cc
    unit/scope.cc
    unit/scoped_allocator.cc
    unit/timing.cc
    unit/logical_array/create_array.cc
    unit/logical_array/delinearize.cc
    unit/logical_array/project.cc
    unit/logical_array/promote.cc
    unit/logical_array/slice.cc
    unit/logical_array/transpose.cc
    unit/logical_array/utils.cc
    unit/logical_store/create_store.cc
    unit/logical_store/delinearize.cc
    unit/logical_store/equal_storage.cc
    unit/logical_store/overlaps.cc
    unit/logical_store/partition.cc
    unit/logical_store/physical_store.cc
    unit/logical_store/project.cc
    unit/logical_store/promote.cc
    unit/logical_store/slice.cc
    unit/logical_store/transpose.cc
    unit/physical_array/create_array.cc
    unit/physical_array/create_list_array.cc
    unit/physical_array/create_string_array.cc
    unit/physical_array/create_struct_array.cc
    unit/physical_array/fill_array.cc
    unit/physical_array/fill_list_array.cc
    unit/physical_array/fill_string_array.cc
    unit/type/binary_type.cc
    unit/type/fixed_array_type.cc
    unit/type/list_type.cc
    unit/type/primitive_type.cc
    unit/type/string_type.cc
    unit/type/struct_type.cc
    unit/type/type_features.cc
    unit/type/type_traits.cc
    # experimental
    experimental/stl/elementwise.cc
    experimental/stl/fill.cc
    experimental/stl/for_each.cc
    experimental/stl/reduce.cc
    experimental/stl/store.cc
    experimental/stl/transform.cc
    experimental/stl/transform_reduce.cc
    experimental/stl/views.cc
    experimental/tracing.cc)

set(wo_runtime_SRC
    noinit/config.cc
    noinit/cuda_driver_api.cc
    noinit/enable_shared_from_this.cc
    noinit/enumerate.cc
    noinit/environment_variable.cc
    noinit/find_memory_kind.cc
    noinit/internal_shared_ptr.cc
    noinit/internal_weak_ptr.cc
    noinit/is_running_in_task.cc
    noinit/macros.cc
    noinit/runtime_exception.cc
    noinit/scope_fail.cc
    noinit/scope_guard.cc
    noinit/shared_ptr.cc
    noinit/span.cc
    noinit/task_exception.cc
    noinit/to_domain.cc
    noinit/tuple.cc
    noinit/variant_options.cc
    noinit/zip_equal.cc
    noinit/zip_shortest.cc
    noinit/delinearize.cc)

set(non_reentrant_with_runtime_SRC
    non_reentrant/death_example.cc non_reentrant/oom_message.cc non_reentrant/runtime.cc
    non_reentrant/shutdown_callback.cc)

set(non_reentrant_wo_runtime_SRC
    non_reentrant/wo_runtime/exception/base.cc
    non_reentrant/wo_runtime/exception/traced_exception.cc
    non_reentrant/wo_runtime/exception/terminate_handler.cc)

legate_configure_test(NAME tests_with_runtime SOURCES main_with_runtime_init.cc
                                                      ${with_runtime_SRC})
legate_configure_test(NAME tests_wo_runtime SOURCES main_wo_runtime_init.cc
                                                    ${wo_runtime_SRC})
legate_configure_test(NAME tests_non_reentrant_with_runtime
                      SOURCES main_with_runtime_init.cc ${non_reentrant_with_runtime_SRC})
legate_configure_test(NAME tests_non_reentrant_wo_runtime
                      SOURCES main_wo_runtime_init.cc ${non_reentrant_wo_runtime_SRC})

if(Legion_USE_CUDA)
  set(gpu_SRC integration/nccl.cu)

  legate_configure_test(NAME tests_with_gpus SOURCES main_with_runtime_init.cc
                                                     ${with_runtime_SRC} ${gpu_SRC} GPUS 1
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
