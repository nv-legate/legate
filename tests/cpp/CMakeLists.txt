#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)

project(legate_tests VERSION 0.1 LANGUAGES C CXX)

if(PROJECT_IS_TOP_LEVEL)
  # To catch people trying to build the tests from within tests/cpp instead of top-level
  message(
    FATAL_ERROR
    "Error: Tests can only be built as part of the main library build. Please re-run cmake from top-level directory (\${LEGATE_DIR}) with -Dlegate_BUILD_TESTS=ON"
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
  set(multi_value SOURCES ADDITIONAL_TARGETS)
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

  add_executable(
    ${_LEGATE_TEST_NAME}
    ${_LEGATE_TEST_SOURCES}
    utilities/sanitizer_defaults.cc
  )

  set_target_properties(
    ${_LEGATE_TEST_NAME}
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}>"
      BUILD_RPATH_USE_ORIGIN TRUE
      INSTALL_RPATH_USE_LINK_PATH TRUE
      LEGATE_INTERNAL_TARGET TRUE
  )

  set(
    install_rpaths
    "${legate_PLATFORM_RPATH_ORIGIN}/../${CMAKE_INSTALL_LIBDIR}"
    "${legate_PLATFORM_RPATH_ORIGIN}/../${legate_DEP_INSTALL_LIBDIR}"
  )
  set_property(
    TARGET ${_LEGATE_TEST_NAME}
    APPEND
    PROPERTY INSTALL_RPATH "${install_rpaths}"
  )

  target_include_directories(${_LEGATE_TEST_NAME} PRIVATE ${PROJECT_SOURCE_DIR})

  target_link_libraries(
    ${_LEGATE_TEST_NAME}
    PRIVATE
      legate_obj
      GTest::gtest
      GTest::gmock
      # Some test modules require delving into the private
      # headers of legate, which may contain fmt headers. As a
      # result, we also need to add the fmt include paths etc.
      # This is not done automatically for us, since fmt is a
      # PRIVATE dependency of legate (as it is not found in any
      # public includes).
      fmt::fmt-header-only
      ${_LEGATE_TEST_ADDITIONAL_TARGETS}
  )

  rapids_test_add(
    NAME ${_LEGATE_TEST_NAME}
    COMMAND ${_LEGATE_TEST_NAME}
    GPUS ${_LEGATE_TEST_GPUS}
    PERCENT ${_LEGATE_TEST_PERCENT}
    INSTALL_COMPONENT_SET testing
  )

  # TODO(jfaibussowit): We need to use DISCOVERY_MODE PRE_TEST (as described below), but
  # CMake has a bug in their implementation. See
  # https://gitlab.kitware.com/cmake/cmake/-/issues/26294
  #
  # gtest_discover_tests(${_LEGATE_TEST_NAME}
  #                      PROPERTIES ENVIRONMENT
  #                                 LEGATE_TEST=1
  #                                 # Cannot discover tests at build time because CI splits
  #                                 # build and run into 2 separate phase. The build phase
  #                                 # does not necessarily have the required libs to run the
  #                                 # execute.
  #                                 DISCOVERY_MODE
  #                                 PRE_TEST)

  legate_install_debug_symbols(
    TARGET ${_LEGATE_TEST_NAME}
    INSTALL_DIR ${CMAKE_INSTALL_BINDIR}
  )

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

set(with_runtime_TARGETS)
set(
  with_runtime_SRC
  # bug
  bug/alias_via_promote.cc
  bug/aligned_unpack.cc
  bug/dangling_store_partition.cc
  bug/key_partition_invalidation.cc
  bug/logical_store_transform.cc
  bug/nullable_arrays.cc
  bug/premature_free.cc
  bug/remap_double_detach.cc
  bug/singleton_index_task.cc
  bug/projection_c_order.cc
  bug/transpose_c_order.cc
  bug/transpose_fortran_order.cc
  bug/temporary_logical_store.cc
  bug/string_physical_array_owner.cc
  bug/private_scalar_copy.cc
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
  integration/core_mapper.cc
  integration/copy_gather_scatter.cc
  integration/copy_normal.cc
  integration/copy_scatter.cc
  integration/delinearize.cc
  integration/exception.cc
  integration/field_reuse.cc
  integration/fill.cc
  integration/find_memory_kind.cc
  integration/image_constraints.cc
  integration/index_attach.cc
  integration/inline_map.cc
  integration/input_output.cc
  integration/is_partitioned.cc
  integration/is_running_in_task.cc
  integration/machine_scope.cc
  integration/manual_task.cc
  integration/manual_task_proj.cc
  integration/mapping.cc
  integration/mixed_dim.cc
  integration/multi_scalar_out.cc
  integration/partitioner.cc
  integration/physical_task.cc
  integration/projection.cc
  integration/proc_local_storage.cc
  integration/provenance.cc
  integration/redundant.cc
  integration/region_manager.cc
  integration/register_variants.cc
  integration/replicated_write.cc
  integration/req_analyzer.cc
  integration/scalar_out.cc
  integration/scale_constraints.cc
  integration/store_colocation.cc
  integration/streaming_checker.cc
  integration/task_misc.cc
  integration/comm/cpu_allreduce_typed.cc
  integration/comm/cpu_communicator.cc
  integration/comm/cpu_comm_exception.cc
  integration/tasks/task_simple.cc
  integration/task_store/auto_task_tests.cc
  integration/task_store/manual_task_tests.cc
  integration/task_store/physical_task/physical_task_basic_tests.cc
  integration/task_store/physical_task/physical_task_context_tests.cc
  integration/task_store/physical_task/physical_task_nested_tests.cc
  integration/task_store/physical_task/physical_task_error_tests.cc
  integration/transpose_dim_order.cc
  integration/tree_reduce.cc
  integration/tree_reduce_unique.cc
  integration/tunable.cc
  integration/variant_options_precedence.cc
  integration/opaque.cc
  # unit
  unit/attachment.cc
  unit/constraint.cc
  unit/data/partition_placement_info.cc
  unit/dispatch.cc
  unit/formatter.cc
  unit/future_wrapper.cc
  unit/library.cc
  unit/logical_region_field.cc
  unit/registration.cc
  unit/runtime.cc
  unit/scope.cc
  unit/scoped_allocator.cc
  unit/shape.cc
  unit/storage.cc
  unit/timing.cc
  unit/buffer/typed_buffer.cc
  unit/buffer/task_local_buffer.cc
  unit/logical_array/create_list_array.cc
  unit/logical_array/create_primitive_array.cc
  unit/logical_array/create_string_array.cc
  unit/logical_array/create_struct_array.cc
  unit/logical_array/delinearize.cc
  unit/logical_array/project.cc
  unit/logical_array/promote.cc
  unit/logical_array/slice.cc
  unit/logical_array/transpose.cc
  unit/logical_array/utils.cc
  unit/logical_store/broadcast.cc
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
  unit/logical_store/reinterpret_as.cc
  unit/machine/node_range.cc
  unit/machine/processor_range.cc
  unit/machine/machine.cc
  unit/machine/global_machine.cc
  unit/machine/local_machine.cc
  unit/machine/local_machine_selector.cc
  unit/mapping/array/base_array.cc
  unit/mapping/array/list_array.cc
  unit/mapping/array/struct_array.cc
  unit/mapping/common_api.cc
  unit/mapping/dim_ordering.cc
  unit/mapping/instance_manager.cc
  unit/mapping/instance_mapping_policy.cc
  unit/mapping/store/api_test.cc
  unit/mapping/store/colocate.cc
  unit/mapping/store/domain.cc
  unit/mapping/store/future_wrapper.cc
  unit/mapping/store/properties.cc
  unit/mapping/store/transform.cc
  unit/mapping/store_mapping.cc
  unit/partition/image.cc
  unit/partition/nopartition.cc
  unit/partition/tiling.cc
  unit/partition/opaque.cc
  unit/physical_array/create_array.cc
  unit/physical_array/create_list_array.cc
  unit/physical_array/create_string_array.cc
  unit/physical_array/create_struct_array.cc
  unit/physical_array/fill_array.cc
  unit/physical_array/fill_list_array.cc
  unit/physical_array/fill_string_array.cc
  unit/physical_store/create_bound_store.cc
  unit/physical_store/create_by_physical_array.cc
  unit/physical_store/create_future_store.cc
  unit/physical_store/create_unbound_store.cc
  unit/physical_store/inline_allocation.cc
  unit/physical_store/read_accessor.cc
  unit/physical_store/read_write_accessor.cc
  unit/physical_store/reduce_accessor.cc
  unit/physical_store/transform.cc
  unit/physical_store/write_accessor.cc
  unit/redop/apply.cc
  unit/redop/complex.cc
  unit/redop/fold.cc
  unit/redop/half.cc
  unit/redop/identity.cc
  unit/scalar/create_primitive_scalar.cc
  unit/scalar/create_struct_scalar.cc
  unit/scalar/create_vector_scalar.cc
  unit/scalar/negative.cc
  unit/scalar/pack.cc
  unit/transform/broadcast.cc
  unit/transform/delinearize.cc
  unit/transform/project.cc
  unit/transform/promote.cc
  unit/transform/shift.cc
  unit/transform/transform_stack.cc
  unit/transform/transpose.cc
  unit/type/binary_type.cc
  unit/type/fixed_array_type.cc
  unit/type/list_type.cc
  unit/type/primitive_type.cc
  unit/type/string_type.cc
  unit/type/struct_type.cc
  unit/type/type_features.cc
  unit/type/type_traits.cc
  unit/task_signature/nargs.cc
  unit/task_signature/basic.cc
  unit/task_signature/register.cc
  unit/task_signature/alignment.cc
  unit/task_signature/bloat.cc
  unit/task_signature/broadcast.cc
  unit/task_signature/image.cc
  unit/task_signature/scale.cc
  unit/task_config/basic.cc
  unit/dlpack/from_dlpack/versioned.cc
  unit/dlpack/from_dlpack/unversioned.cc
  unit/dlpack/to_dlpack/to_dlpack.cc
  unit/streaming/extract_streamable_prefix.cc
  unit/streaming/nested_scopes.cc
  unit/streaming/process_streaming_run.cc
  unit/task/physical_task.cc
  # experimental
  experimental/stl/elementwise.cc
  experimental/stl/fill.cc
  experimental/stl/for_each.cc
  experimental/stl/reduce.cc
  experimental/stl/store.cc
  experimental/stl/transform.cc
  experimental/stl/transform_reduce.cc
  experimental/stl/views.cc
  experimental/tracing.cc
)

if(legate_USE_HDF5)
  list(
    APPEND with_runtime_SRC
    unit/io/hdf5/negative.cc
    unit/io/hdf5/read.cc
    unit/io/hdf5/write.cc
    unit/io/hdf5/partitioner.cc
  )
  list(APPEND with_runtime_TARGETS HDF5::HDF5)
endif()

set(
  wo_runtime_SRC
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
  noinit/scope_fail.cc
  noinit/scope_guard.cc
  noinit/shared_ptr.cc
  noinit/task_exception.cc
  noinit/to_domain.cc
  noinit/tuple.cc
  noinit/variant_options.cc
  noinit/zip_equal.cc
  noinit/zip_shortest.cc
  noinit/delinearize.cc
  noinit/unravel.cc
  noinit/argument_parsing/parse_args.cc
  noinit/argument_parsing/gpus.cc
  noinit/argument_parsing/fbmem.cc
  noinit/argument_parsing/omps.cc
  noinit/argument_parsing/numamem.cc
  noinit/argument_parsing/sysmem.cc
  noinit/argument_parsing/cpus.cc
  noinit/argument_parsing/ompthreads.cc
  noinit/argument_parsing/logging.cc
  noinit/argument_parsing/legion_default_args.cc
  noinit/argument_parsing/argument.cc
  noinit/argument_parsing/scaled.cc
  noinit/argument_parsing/cuda_driver_path.cc
)

set(wo_runtime_EXTRA_TARGETS argparse::argparse)

if(legate_USE_UCX)
  list(
    APPEND wo_runtime_SRC
    integration/comm/ucc_integration.cc
    unit/comm/ucc_network.cc
    unit/comm/oob_allgather.cc
  )
  list(APPEND wo_runtime_EXTRA_TARGETS ucx::ucp ucx::ucs ucc::ucc)
endif()

set(
  non_reentrant_with_runtime_SRC
  non_reentrant/death_example.cc
  unit/logical_store/offload_to.cc
  unit/logical_array/offload_to.cc
  non_reentrant/oom_message.cc
  non_reentrant/runtime.cc
  non_reentrant/shutdown_callback.cc
  non_reentrant/sharding.cc
)

set(
  non_reentrant_wo_runtime_SRC
  non_reentrant/wo_runtime/exception/base.cc
  non_reentrant/wo_runtime/exception/traced_exception.cc
  non_reentrant/wo_runtime/exception/terminate_handler.cc
  non_reentrant/wo_runtime/init/init.cc
  non_reentrant/wo_runtime/streaming/streaming.cc
  non_reentrant/wo_runtime/inline_launch/basic.cc
  non_reentrant/wo_runtime/comm/backend_network.cc
  non_reentrant/wo_runtime/comm/coll_abort.cc
)

legate_configure_test(
  NAME tests_with_runtime
  SOURCES main_with_runtime_init.cc ${with_runtime_SRC}
  ADDITIONAL_TARGETS ${with_runtime_TARGETS}
)
legate_configure_test(
  NAME tests_wo_runtime
  SOURCES main_wo_runtime_init.cc ${wo_runtime_SRC}
  ADDITIONAL_TARGETS ${wo_runtime_EXTRA_TARGETS}
)
legate_configure_test(
  NAME tests_non_reentrant_with_runtime
  SOURCES main_with_runtime_init.cc ${non_reentrant_with_runtime_SRC}
)
legate_configure_test(
  NAME tests_non_reentrant_wo_runtime
  SOURCES main_wo_runtime_init.cc ${non_reentrant_wo_runtime_SRC}
)

set(gpu_SRC)
set(gpu_TARGETS)

if(legate_USE_NCCL)
  list(APPEND gpu_SRC integration/nccl.cu)
  list(APPEND gpu_TARGETS NCCL::NCCL)
endif()

if(legate_USE_CUDA)
  legate_configure_test(
    NAME tests_with_gpus
    SOURCES main_with_runtime_init.cc ${with_runtime_SRC} ${gpu_SRC}
    GPUS 1
    PERCENT 30
    ADDITIONAL_TARGETS ${with_runtime_TARGETS} ${gpu_TARGETS}
  )
endif()

rapids_test_install_relocatable(
  INSTALL_COMPONENT_SET testing
  DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDE_IN_ALL
)

function(legate_create_dummy_cuda_driver)
  # This library MUST be a shared library for it to work properly
  add_library(legate_dummy_cuda_driver SHARED utilities/dummy_cuda_driver.cc)
  set_target_properties(
    legate_dummy_cuda_driver
    PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_INSTALL_LIBDIR}
  )
  # tests_wo_runtime contains the CUDADriverAPI test which tests whether we can
  # successfully load a shim CUDA driver module. By linking the library into
  # tests_wo_runtime, it adds it to the RPATH, we can just pass
  # "liblegate_dummy_cuda_driver.so" to the loader, and because its in the RPATH, dlopen()
  # will find it.
  #
  # This works because dir structure is like so:
  #
  # CMAKE_CURRENT_BINARY_DIR/
  # | - bin/
  # |   | - tests_with_runtime
  # | - lib/
  #     | - liblegate_dummy_cuda_driver.dylib
  #
  # As an added bonus, CMake will also ensure this rpath gets updated when both of these
  # get installed.
  target_link_libraries(tests_wo_runtime PRIVATE legate_dummy_cuda_driver)
  install(TARGETS legate_dummy_cuda_driver DESTINATION ${CMAKE_INSTALL_LIBDIR})
endfunction()
legate_create_dummy_cuda_driver()
