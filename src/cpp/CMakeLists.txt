#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

list(APPEND CMAKE_MESSAGE_CONTEXT "cpp")

# ########################################################################################
# * User Options  ------------------------------------------------------------

include("${LEGATE_CMAKE_DIR}/Modules/legate_options.cmake")

# ########################################################################################
# * Project definition -------------------------------------------------------

# Write the version header
rapids_cmake_write_version_file(${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/version.h
                                PREFIX LEGATE)

if(PROJECT_IS_TOP_LEVEL)
  message(STATUS "Symlinking compile_commands.json to root directory")
  file(CREATE_LINK
       # Using CMAKE_BINARY_DIR vs CMAKE_CURRENT_BINARY_DIR is intentional. If we use
       # CURRENT_BINARY_DIR, then it breaks anytime someone uses add_subdirectory(), since
       # cmake always puts the compile_commands.json in the top-level build dir (not in
       # the sub tree)
       "${CMAKE_BINARY_DIR}/compile_commands.json"
       "${LEGATE_DIR}/compile_commands.json"
       SYMBOLIC)
endif()

# ########################################################################################
# * Dependencies -------------------------------------------------------------

# add third party dependencies using CPM
rapids_cpm_init(OVERRIDE "${LEGATE_CMAKE_DIR}/versions/versions.json")

# ########################################################################################
# * Default Flags -------------------------------------------------------------

if(legate_USE_CUDA)
  # Must do this here since the debug symbols and default flags need to see if CUDA
  # language support is needed.
  #
  # It is important to define CUDA architectures, as the current default of 52 is too old.
  # This must be populated before the enable_language call as that will set the default.
  if(NOT DEFINED ENV{CUDAARCHS})
    set_ifndef(CMAKE_CUDA_ARCHITECTURES "all-major")
  endif()

  enable_language(CUDA)

  include("${LEGATE_CMAKE_DIR}/Modules/cuda_arch_helpers.cmake")

  # Needs to run before find_package(Legion)
  legate_set_default_cuda_arch(DEST_VAR CMAKE_CUDA_ARCHITECTURES)
  set(Legion_CUDA_ARCH "${CMAKE_CUDA_ARCHITECTURES}")
  set(CMAKE_INCLUDE_SYSTEM_FLAG_CUDA "-isystem ")
  # Find the CUDAToolkit
  rapids_find_package(# Min version of CUDA is 11.8, but we want CMake to prefer the
                      # highest version possible. It seems that it only does that if you
                      # give it a range...
                      #
                      # So in 100 years time, if we ever get to CUDA version 100000000,
                      # someone must remember to add another 9 below.
                      CUDAToolkit 11.8...99999999.99 REQUIRED
                      BUILD_EXPORT_SET legate-exports
                      INSTALL_EXPORT_SET legate-exports)
endif()

include("${LEGATE_CMAKE_DIR}/Modules/default_flags.cmake")

legate_configure_default_compiler_flags()
legate_configure_default_linker_flags()

include("${LEGATE_CMAKE_DIR}/Modules/debug_symbols.cmake")

legate_configure_debug_symbols()

# ########################################################################################
# * Dependencies ---------------------------------------------------------------------

include("${LEGATE_CMAKE_DIR}/Modules/find_or_configure.cmake")

legate_find_or_configure_init()

# ########################################################################################
# * CCCL ---------------------------------------------------------------------

macro(_legate_find_openmp required)
  rapids_find_package(OpenMP ${required}
                      GLOBAL_TARGETS OpenMP::OpenMP_CXX
                      BUILD_EXPORT_SET legate-exports
                      INSTALL_EXPORT_SET legate-exports
                      COMPONENTS CXX)
endmacro()

# Pull CCCL in before Legion, so that Legion will use the same libcu++ as Legate (the one
# pull from CCCL). This also means
if(NOT DEFINED Legion_USE_OpenMP)
  _legate_find_openmp("")
  option(Legion_USE_OpenMP "Use OpenMP for Legion" ${OpenMP_FOUND})
elseif(Legion_USE_OpenMP)
  _legate_find_openmp(REQUIRED)
endif()
# The find_or_configure_cccl() call needs to know whether Legion_USE_OpenMP is true or
# not. We set some thrust variables based on whether Legion_USE_OpenMP is true, which we
# don't really know until we configure Legion. But we must configure CCCL first, in order
# to get Legion to use our CCCL. So chicken and egg problem.
#
# We hack around this by determining whether we want Legion to have OpenMP based on
# whether we find it. But it's not 100% foolproof.
set(legate_checked_for_Legion_USE_OpenMP ON)

legate_find_or_configure(PACKAGE CCCL)

# ########################################################################################
# * Python -------------------------------------------------------------------

macro(_legate_find_python3)
  rapids_find_package(Python3 COMPONENTS Interpreter Development FIND_ARGS REQUIRED)
  message(VERBOSE "legate: Has Python3: ${Python3_FOUND}")
  message(VERBOSE "legate: Has Python 3 interpreter: ${Python3_Interpreter_FOUND}")
  message(VERBOSE "legate: Python 3 include directories: ${Python3_INCLUDE_DIRS}")
  message(VERBOSE "legate: Python 3 libraries: ${Python3_LIBRARIES}")
  message(VERBOSE "legate: Python 3 library directories: ${Python3_LIBRARY_DIRS}")
  message(VERBOSE "legate: Python 3 version: ${Python3_VERSION}")
endmacro()

# ########################################################################################
# * Legion -------------------------------------------------------------------

if(Legion_USE_Python)
  _legate_find_python3()
  if(Python3_FOUND AND Python3_VERSION)
    set(Legion_Python_Version ${Python3_VERSION})
  endif()
endif()

#
# If we find Legion already configured on the system, it will report whether it was
# compiled with Python (Legion_USE_PYTHON), CUDA (Legion_USE_CUDA), OpenMP
# (Legion_USE_OpenMP), and networking (Legion_NETWORKS).
#
# We use the same variables as Legion because we want to enable/disable each of these
# features based on how Legion was configured (it doesn't make sense to build legate's
# Python bindings if Legion's bindings weren't compiled).
#
legate_find_or_configure(PACKAGE Legion)

# If Legion_USE_Python was toggled ON by find_package(Legion), find Python3
if(Legion_USE_Python AND (NOT Python3_FOUND))
  _legate_find_python3()
endif()

if(Legion_USE_OpenMP)
  _legate_find_openmp(REQUIRED)
endif()

# ########################################################################################
# * MPI wrapper --------------------------------------------------------------

if(Legion_NETWORKS AND NOT LEGATE_BUILD_PIP_WHEELS)
  # The wrapper we build should not install anything but the library object itself,
  # because we want to install the sources (and cmake files) to a different location
  # ourselves.
  set(LEGATE_MPI_WRAPPER_SRC_INSTALL_RULES OFF)

  add_subdirectory(${LEGATE_DIR}/share/legate/mpi_wrapper share/legate/mpi_wrapper)
endif()

# ########################################################################################
# * NCCL ----------------------------------------------------------------

if(legate_USE_NCCL)
  legate_find_or_configure(PACKAGE NCCL)
endif()

# ########################################################################################
# * UCX ----------------------------------------------------------------

if(legate_USE_UCX)
  # UCX is not supported on macOS.
  if(APPLE)
    message(FATAL_ERROR "UCX is not supported on macOS. Please disable UCX support.")
  endif()
  legate_find_or_configure(PACKAGE ucx)
  legate_find_or_configure(PACKAGE ucc)
endif()

# ########################################################################################
# * CAL ----------------------------------------------------------------

if(legate_USE_CAL)
  legate_find_or_configure(PACKAGE CAL)
endif()

# ########################################################################################
# * fmt::fmt --------------------------------------------------------------

legate_find_or_configure(PACKAGE fmt)

# ########################################################################################
# * argparse::argparse --------------------------------------------------------------

legate_find_or_configure(PACKAGE argparse)

# ########################################################################################
# * Kvikio --------------------------------------------------------------

legate_find_or_configure(PACKAGE KvikIO)

# ########################################################################################
# * HighFive --------------------------------------------------------------

if(legate_USE_HDF5)
  legate_find_or_configure(PACKAGE HighFive)
endif()

# ########################################################################################
# * HDF5 vfd-gds --------------------------------------------------------------

if(legate_USE_HDF5_VFD_GDS)
  if(NOT legate_USE_HDF5)
    message(FATAL_ERROR "Enabling HDF5 VFD GDS requires HDF5")
  endif()
  legate_find_or_configure(PACKAGE hdf5_vfd_gds)
endif()

# ########################################################################################
# * cpptrace --------------------------------------------------------------

legate_find_or_configure(PACKAGE cpptrace)

# ########################################################################################
# * CPMLicenses --------------------------------------------------------------

legate_find_or_configure(PACKAGE cpm_licenses)

# ########################################################################################
# * legate --------------------------------------------------------------

add_library(legate_obj OBJECT)

# cmake-format: off
set_target_properties(legate_obj
                      PROPERTIES VISIBILITY_INLINES_HIDDEN TRUE
                                 CXX_VISIBILITY_PRESET     "hidden"
                                 CUDA_VISIBILITY_PRESET    "hidden"
                                 DEFINE_SYMBOL             "legate_EXPORTS")
# cmake-format: on

# Define this property here because it directly relates to the legate target. Targets
# linking against legate, which have this property defined, will also inherit the private
# flags of Legate. This is used to propagate these in a selective manner. For example, the
# tests, Cython bindings, and the benchmarks all have this property defined.
#
# Use of this property, however, is brittle, and it is very easy to forget to set this
# property where it should be. For that reason, we define the property here, and also
# check its existence on a few targets which we know must have it. That is the best we can
# do...
define_property(TARGET PROPERTY LEGATE_INTERNAL_TARGET
                BRIEF_DOCS "A boolean property indicating that a target is internal to "
                           "the legate build")

include("${LEGATE_CMAKE_DIR}/Modules/generate_sanitizer_options.cmake")

legate_generate_sanitizer_options(
  SRC ${LEGATE_DIR}/share/legate/sanitizers/asan_default_options.txt
  DELIM [[:]]
  DEST "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/generated/asan_default_options.h"
)

legate_generate_sanitizer_options(
  SRC ${LEGATE_DIR}/share/legate/sanitizers/lsan_suppressions.txt
  DELIM [[\n]]
  DEST "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/generated/lsan_suppressions.h"
)

legate_generate_sanitizer_options(
  SRC ${LEGATE_DIR}/share/legate/sanitizers/ubsan_default_options.txt
  DELIM [[:]]
  DEST "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/generated/ubsan_default_options.h"
)
legate_generate_sanitizer_options(
  SRC ${LEGATE_DIR}/share/legate/sanitizers/ubsan_suppressions.txt
  DELIM [[\n]]
  DEST "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/generated/ubsan_suppressions.h"
)

legate_generate_sanitizer_options(
  SRC ${LEGATE_DIR}/share/legate/sanitizers/tsan_suppressions.txt
  DELIM [[\n]]
  DEST "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/generated/tsan_suppressions.h"
)
legate_generate_sanitizer_options(
  SRC ${LEGATE_DIR}/share/legate/sanitizers/tsan_default_options.txt
  DELIM [[:]]
  DEST "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/generated/tsan_default_options.h"
)

target_sources(legate_obj
               PRIVATE # core
                       legate/comm/coll.cc
                       legate/comm/detail/coll.cc
                       legate/comm/detail/backend_network.cc
                       legate/comm/detail/comm.cc
                       legate/comm/detail/comm_cpu.cc
                       legate/comm/detail/comm_local.cc
                       legate/comm/detail/local_network.cc
                       legate/comm/detail/logger.cc
                       legate/comm/detail/thread_comm.cc
                       legate/cuda/stream_pool.cc
                       legate/cuda/detail/cuda_driver_api.cc
                       legate/cuda/detail/module_manager.cc
                       legate/data/allocator.cc
                       legate/data/buffer.cc
                       legate/data/external_allocation.cc
                       legate/data/logical_array.cc
                       legate/data/logical_store.cc
                       legate/data/scalar.cc
                       legate/data/shape.cc
                       legate/data/physical_array.cc
                       legate/data/physical_store.cc
                       legate/data/detail/array_tasks.cc
                       legate/data/detail/attachment.cc
                       legate/data/detail/external_allocation.cc
                       legate/data/detail/logical_array.cc
                       legate/data/detail/logical_region_field.cc
                       legate/data/detail/logical_store.cc
                       legate/data/detail/scalar.cc
                       legate/data/detail/physical_array.cc
                       legate/data/detail/physical_store.cc
                       legate/data/detail/shape.cc
                       legate/data/detail/transform.cc
                       legate/data/detail/future_wrapper.cc
                       legate/data/detail/region_field.cc
                       legate/data/detail/user_storage_tracker.cc
                       legate/data/detail/buffer.cc
                       legate/experimental/trace.cc
                       legate/mapping/array.cc
                       legate/mapping/machine.cc
                       legate/mapping/mapping.cc
                       legate/mapping/operation.cc
                       legate/mapping/store.cc
                       legate/mapping/detail/array.cc
                       legate/mapping/detail/base_mapper.cc
                       legate/mapping/detail/core_mapper.cc
                       legate/mapping/detail/instance_manager.cc
                       legate/mapping/detail/machine.cc
                       legate/mapping/detail/mapping.cc
                       legate/mapping/detail/operation.cc
                       legate/mapping/detail/store.cc
                       legate/operation/projection.cc
                       legate/operation/task.cc
                       legate/operation/detail/attach.cc
                       legate/operation/detail/copy.cc
                       legate/operation/detail/copy_launcher.cc
                       legate/operation/detail/discard.cc
                       legate/operation/detail/execution_fence.cc
                       legate/operation/detail/fill.cc
                       legate/operation/detail/fill_launcher.cc
                       legate/operation/detail/gather.cc
                       legate/operation/detail/index_attach.cc
                       legate/operation/detail/mapping_fence.cc
                       legate/operation/detail/launcher_arg.cc
                       legate/operation/detail/operation.cc
                       legate/operation/detail/store_projection.cc
                       legate/operation/detail/reduce.cc
                       legate/operation/detail/release_region_field.cc
                       legate/operation/detail/store_analyzer.cc
                       legate/operation/detail/scatter.cc
                       legate/operation/detail/scatter_gather.cc
                       legate/operation/detail/task.cc
                       legate/operation/detail/task_launcher.cc
                       legate/operation/detail/timing.cc
                       legate/partitioning/constraint.cc
                       legate/partitioning/proxy.cc
                       legate/partitioning/detail/constraint.cc
                       legate/partitioning/detail/constraint_solver.cc
                       legate/partitioning/detail/partition.cc
                       legate/partitioning/detail/partitioner.cc
                       legate/partitioning/detail/partitioning_tasks.cc
                       legate/partitioning/detail/restriction.cc
                       legate/partitioning/detail/strategy.cc
                       legate/partitioning/detail/proxy/align.cc
                       legate/partitioning/detail/proxy/broadcast.cc
                       legate/partitioning/detail/proxy/image.cc
                       legate/partitioning/detail/proxy/scale.cc
                       legate/partitioning/detail/proxy/bloat.cc
                       legate/partitioning/detail/proxy/validate.cc
                       legate/runtime/library.cc
                       legate/runtime/runtime.cc
                       legate/runtime/detail/communicator_manager.cc
                       legate/runtime/detail/field_manager.cc
                       legate/runtime/detail/library.cc
                       legate/runtime/detail/partition_manager.cc
                       legate/runtime/detail/projection.cc
                       legate/runtime/detail/region_manager.cc
                       legate/runtime/detail/runtime.cc
                       legate/runtime/detail/shard.cc
                       legate/runtime/detail/mapper_manager.cc
                       legate/runtime/detail/argument_parsing/util.cc
                       legate/runtime/detail/scope.cc
                       legate/runtime/detail/argument_parsing/parse.cc
                       legate/runtime/detail/argument_parsing/config_legion.cc
                       legate/runtime/detail/argument_parsing/config_realm.cc
                       legate/runtime/detail/argument_parsing/exceptions.cc
                       legate/runtime/detail/argument_parsing/legate_args.cc
                       legate/runtime/detail/argument_parsing/flags/cpus.cc
                       legate/runtime/detail/argument_parsing/flags/gpus.cc
                       legate/runtime/detail/argument_parsing/flags/fbmem.cc
                       legate/runtime/detail/argument_parsing/flags/logging.cc
                       legate/runtime/detail/argument_parsing/flags/numamem.cc
                       legate/runtime/detail/argument_parsing/flags/ompthreads.cc
                       legate/runtime/detail/argument_parsing/flags/openmp.cc
                       legate/runtime/detail/argument_parsing/flags/sysmem.cc
                       legate/runtime/detail/argument_parsing/flags/cuda_driver_path.cc
                       legate/runtime/detail/mpi_detection.cc
                       legate/runtime/detail/streaming.cc
                       legate/task/registrar.cc
                       legate/task/task.cc
                       legate/task/task_context.cc
                       legate/task/task_info.cc
                       legate/task/variant_options.cc
                       legate/task/variant_info.cc
                       legate/task/task_signature.cc
                       legate/task/task_config.cc
                       legate/task/detail/return_value.cc
                       legate/task/detail/returned_exception.cc
                       legate/task/detail/returned_cpp_exception.cc
                       legate/task/detail/returned_python_exception.cc
                       legate/task/detail/task_context.cc
                       legate/task/detail/inline_task_body.cc
                       legate/task/detail/legion_task_body.cc
                       legate/task/detail/task.cc
                       legate/task/detail/task_return.cc
                       legate/task/detail/task_return_layout.cc
                       legate/task/detail/variant_info.cc
                       legate/task/detail/task_info.cc
                       legate/tuning/parallel_policy.cc
                       legate/tuning/scope.cc
                       legate/task/detail/task_signature.cc
                       legate/task/detail/task_config.cc
                       legate/type/types.cc
                       legate/type/detail/types.cc
                       legate/utilities/machine.cc
                       legate/utilities/internal_shared_ptr.cc
                       legate/utilities/compiler.cc
                       legate/utilities/abort.cc
                       legate/utilities/dispatch.cc
                       legate/utilities/detail/align.cc
                       legate/utilities/detail/buffer_builder.cc
                       legate/utilities/detail/env.cc
                       legate/utilities/detail/tuple.cc
                       legate/utilities/detail/deserializer.cc
                       legate/utilities/detail/formatters.cc
                       legate/utilities/detail/traced_exception.cc
                       legate/utilities/detail/zip.cc
                       legate/utilities/detail/linearize.cc
                       legate/utilities/detail/error.cc
                       legate/utilities/detail/proc_local_storage.cc
                       legate/utilities/detail/array_algorithms.cc
                       legate/utilities/detail/dlpack/from_dlpack.cc
                       legate/utilities/detail/dlpack/to_dlpack.cc
                       legate/utilities/detail/dlpack/common.cc
                       legate/timing/timing.cc
                       # stl
                       legate/experimental/stl/detail/clang_tidy_dummy.cpp
                       # io
                       legate/io/hdf5/interface.cc
                       legate/experimental/io/kvikio/detail/basic.cc
                       legate/experimental/io/kvikio/detail/tile.cc
                       legate/experimental/io/kvikio/detail/tile_by_offsets.cc
                       legate/experimental/io/kvikio/interface.cc
                       legate/experimental/io/detail/task.cc
                       legate/experimental/io/detail/library.cc
                       legate/experimental/io/detail/mapper.cc)

if(legate_USE_HDF5)
  target_sources(legate_obj
                 PRIVATE legate/io/hdf5/detail/interface.cc legate/io/hdf5/detail/read.cc
                         legate/io/hdf5/detail/util.cc)
endif()

if(legate_ENABLE_SANITIZERS)
  target_sources(legate_obj PRIVATE legate/utilities/detail/sanitizer_defaults.cc)
endif()

if(Legion_NETWORKS)
  target_sources(legate_obj
                 PRIVATE legate/comm/detail/mpi_network.cc
                         legate/comm/detail/mpi_interface.cc
                         legate/comm/detail/comm_mpi.cc)
  if(NOT TARGET legate::mpi_wrapper)
    # The MPI wrapper's headers are needed for some symbols, no link.
    target_include_directories(legate_obj
                               PRIVATE "${legate_cpp_SOURCE_DIR}/../share/legate/mpi_wrapper/src"
    )
    # This is needed to tell mpi_wrapper.h not to
    #
    # #include <legate_mpi_wrapper_exports.h>
    #
    # Normally, we need to ensure that the MPI wrappers symbols are publicly visible, but
    # in this case when we just need the header to define the symbols for us (and never
    # intend to actually load the wrapper), we don't care about visibility.
    target_compile_definitions(legate_obj
                               PRIVATE LEGATE_MPI_WRAPPER_HAVE_NO_EXPORT_HEADER=1)
  endif()
endif()

if(Legion_USE_OpenMP)
  target_sources(legate_obj PRIVATE legate/data/detail/array_tasks_omp.cc
                                    legate/partitioning/detail/partitioning_tasks_omp.cc)
endif()

if(legate_USE_CAL)
  target_sources(legate_obj PRIVATE legate/comm/detail/comm_cal.cc)
endif()

if(legate_USE_NCCL)
  target_sources(legate_obj PRIVATE legate/comm/detail/comm_nccl.cc)
endif()

if(legate_USE_CUDA)
  target_sources(legate_obj PRIVATE legate/data/detail/array_tasks_cuda.cc
                                    legate/partitioning/detail/partitioning_tasks_cuda.cc)

  include("${LEGATE_CMAKE_DIR}/Modules/generate_fatbin_modules.cmake")

  legate_generate_fatbin_modules(DEST_DIR "${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/generated/fatbin"
                                 GENERATED_SOURCES_VAR fatbin_src
                                 SOURCES legate/data/detail/fixup_ranges_fatbin.cu
                                         legate/data/detail/offsets_to_ranges_fatbin.cu
                                         legate/data/detail/ranges_to_offsets_fatbin.cu
                                         legate/partitioning/detail/partitioning_tasks_fatbin.cu
                                 EXTRA_FLAGS ${legate_CUDA_FLAGS})

  target_sources(legate_obj PRIVATE ${fatbin_src})
endif()

include("${LEGATE_CMAKE_DIR}/Modules/generate_git_revision_file.cmake")

legate_generate_git_revision_file(
  GENERATED_SRC_VAR git_revision_src EXTRA_TARGETS Legion::Legion
  EXTRA_TARGETS_PREFIX LEGION)

target_sources(legate_obj PRIVATE ${git_revision_src})

include(GenerateExportHeader)

# cmake-format doesn't understand block() (because it hasn't been updated in over 4
# years), and so doesn't indent these correctly.
block()
set(extra
    [=[
// For symbols that are only exported because they are used by the python bindings, not
// because they otherwise need to be. If references to them are removed in the Python
// bindings, these symbols should also be fixed up to be un-exported again.
#define LEGATE_PYTHON_EXPORT LEGATE_EXPORT
]=])
set(export_header
    "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/legate_exports.h"
)

# cmake-format: off
generate_export_header(
  legate_obj
    BASE_NAME                   LEGATE
    EXPORT_FILE_NAME            "${export_header}"
    CUSTOM_CONTENT_FROM_VARIABLE extra
)
# cmake-format: on
endblock()

include("${LEGATE_CMAKE_DIR}/Modules/clang_tidy.cmake")

legate_add_tidy_diff_target()

get_target_property(legate_src legate_obj SOURCES)
foreach(src IN LISTS legate_src)
  legate_add_tidy_target(SOURCE "${src}")
endforeach()

include("${LEGATE_CMAKE_DIR}/Modules/generate_legate_defines.cmake")

legate_generate_legate_defines()

# ########################################################################################
# * Create the Legate target proper ------------------------------------------------------

add_library(legate)
add_library(legate::legate ALIAS legate)

target_link_libraries(legate PRIVATE legate_obj)

# For whatever reason cmake-format craps out when formatting the below, so much so that
# cmake-lint then ultimately complains with:
#
# cmake-format: off
# src/cpp/CMakeLists.txt:480,80: [C0307] Bad indentation:
#                                                              )
# ^----BodyNode: 1:0->StatementNode: 477:0->TreeNode: 480:80
# cmake-format: on
# cmake-format: off
set_target_properties(legate
                        PROPERTIES
                          EXPORT_NAME               legate
                          LIBRARY_OUTPUT_NAME       legate
                          LIBRARY_OUTPUT_DIRECTORY  lib
                          BUILD_RPATH_USE_ORIGIN    TRUE
                          SOVERSION                 ${PROJECT_VERSION}
                          VISIBILITY_INLINES_HIDDEN TRUE
                          CXX_VISIBILITY_PRESET     "hidden"
                          CUDA_VISIBILITY_PRESET    "hidden")
# cmake-format: on

set(install_rpath
    "${legate_PLATFORM_RPATH_ORIGIN}"
    # The ../ is needed because
    # ${legate_PLATFORM_RPATH_ORIGIN}/${legate_DEP_INSTALL_LIBDIR} expands to (on linux)
    # $ORIGIN/lib/legate/deps, which is fine *except* if $ORIGIN is already in lib/. In
    # that case it would expand to lib/lib/legate/deps which does not exist. So we need to
    # ../ once so that it expands to lib/../lib/legate/deps. A hack for sure, but it
    # works.
    "${legate_PLATFORM_RPATH_ORIGIN}/../${legate_DEP_INSTALL_LIBDIR}")

set_property(TARGET legate APPEND PROPERTY INSTALL_RPATH "${install_rpath}")
# export this so that install_info.py can properly locate the versioned and unversioned
# library names
set_property(TARGET legate APPEND PROPERTY EXPORT_PROPERTIES LIBRARY_OUTPUT_NAME)

# ########################################################################################
# * Common Custom User Flags Applied to both legate and legate_obj -----------------------

include("${LEGATE_CMAKE_DIR}/Modules/set_cpu_arch_flags.cmake")

set_cpu_arch_flags(legate_CXX_FLAGS)

# The following flags, linked targets and other such "public" attributes need to be set
# twice, both on the object library and legate itself. The reason for this is that legate
# links to the object library privately, and so any transitive dependencies or flags don't
# bubble up to any downstream libraries.
foreach(target legate_obj legate)
  target_link_libraries("${target}"
                        # Order is important here. We want conda includes to go last
                        # because the conda env might contain other versions of the
                        # libraries below (for example CCCL). We want the conda includes
                        # to go last so that we ensure that compiler pick up the right
                        # headers.
                        #
                        # This is also why CCCL::Thrust comes *before* Legion, because
                        # Legion may pick up headers from inside the conda environment.
                        #
                        # Similarly, we want the MPI wrapper to load as early as possible,
                        # since it provides our MPI symbols.
                        PUBLIC $<TARGET_NAME_IF_EXISTS:legate::mpi_wrapper> #
                               CCCL::CCCL #
                               Legion::Legion #
                               $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
                        PRIVATE $<TARGET_NAME_IF_EXISTS:NCCL::NCCL>
                                # See
                                # https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html#nvtx3
                                $<TARGET_NAME_IF_EXISTS:CUDA::nvtx3>
                                $<COMPILE_ONLY:fmt::fmt-header-only>
                                $<COMPILE_ONLY:argparse::argparse>
                                kvikio::kvikio
                                $<TARGET_NAME_IF_EXISTS:hdf5_vfd_gds>
                                $<TARGET_NAME_IF_EXISTS:HighFive::HighFive>
                                cpptrace::cpptrace
                                $<TARGET_NAME_IF_EXISTS:CAL::CAL>)
  if(legate_USE_UCX)
    target_link_libraries("${target}" PRIVATE ucx::ucp ucx::ucs ucc::ucc)
  endif()
endforeach()

target_compile_definitions(legate_obj PUBLIC $<BUILD_INTERFACE:legate_EXPORTS=1>)
target_compile_definitions(legate PUBLIC $<BUILD_INTERFACE:legate_EXPORTS=1>)

include("${LEGATE_CMAKE_DIR}/Modules/utilities.cmake")

legate_add_target_compile_options(legate_obj CXX PRIVATE legate_CXX_FLAGS)
legate_add_target_compile_options(legate CXX PRIVATE legate_CXX_FLAGS)
# Legate has no direct CUDA files that it would compile (all CUDA sources are compiled
# into fatbins and embedded in the library itself). But tests, benchmarks, or other
# sources might. In that case, they would also have the LEGATE_INTERNAL_TARGET property
# set, so let's populate these for that.
legate_add_target_compile_options(legate_obj CUDA PRIVATE legate_CUDA_FLAGS)
legate_add_target_compile_options(legate CUDA PRIVATE legate_CUDA_FLAGS)

legate_add_target_link_options(legate_obj PUBLIC legate_LINKER_FLAGS)
legate_add_target_link_options(legate PUBLIC legate_LINKER_FLAGS)

if(APPLE)
  # See https://github.com/NVIDIA/cccl/pull/2444
  target_compile_definitions(legate_obj PUBLIC _LIBCUDACXX_STRING_H_HAS_CONST_OVERLOADS=1)
  target_compile_definitions(legate PUBLIC _LIBCUDACXX_STRING_H_HAS_CONST_OVERLOADS=1)
endif()

# See https://github.com/nv-legate/legate.internal/pull/1326#issuecomment-2418791698
foreach(target legate_obj legate LegionRuntime)
  target_compile_definitions("${target}" PUBLIC LIBCUDACXX_ENABLE_HOST_NVFP16=1)
  if(legate_USE_CUDA)
    # CCCL 2.7.0 (which is what we use) disables __half support if it detects that it
    # isn't being compiled with a CUDA compiler. I think this is probably a bug, because
    # we always have __half support if you #include <cuda_fp16.h>. So we need to lie to
    # CCCL.
    #
    # See
    # https://github.com/NVIDIA/cccl/blob/v2.7.0/libcudacxx/include/cuda/std/detail/libcxx/include/__config#L919C11-L924
    # and
    # https://github.com/NVIDIA/cccl/blob/v2.7.0/libcudacxx/include/cuda/std/__cccl/compiler.h#L93-L94
    target_compile_definitions("${target}" PUBLIC _LIBCUDACXX_HAS_NVFP16=1)
  endif()
  if(legate_ENABLE_SANITIZERS AND ("${CCCL_VERSION}" VERSION_EQUAL "2.7.0"))
    # /path/to/libcudacxx/../../../include/cuda/std/detail/libcxx/include/__config:1155:12:
    # error: visibility does not match previous declaration 1155 | extern "C"
    # _LIBCUDACXX_HIDE_FROM_ABI void |            ^
    #
    # /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1/__config:419:39:
    # note: expanded from macro '_LIBCPP_EXPORTED_FROM_ABI' 419 | #    define
    # _LIBCPP_EXPORTED_FROM_ABI _LIBCPP_VISIBILITY("default") | ^
    #
    # For __sanitizer_annotate_contiguous_container().
    target_compile_definitions("${target}" PUBLIC _LIBCUDACXX_HAS_NO_ASAN=1)
  endif()
endforeach()

set(legate_LOCAL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

foreach(target legate_obj legate)
  target_include_directories("${target}"
                             PUBLIC $<BUILD_INTERFACE:${legate_LOCAL_INCLUDE_DIR}>
                                    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/legate>
                             INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/legate>
                                       $<INSTALL_INTERFACE:${legate_DEP_INSTALL_INCLUDEDIR}>
  )
endforeach()

# ########################################################################################
# * install targets-----------------------------------------------------------

include(CPack)

install(TARGETS legate DESTINATION ${CMAKE_INSTALL_LIBDIR} EXPORT legate-exports)

install(FILES legate.h
              ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/legate/legate_defines.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/version.h
              ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/legate_exports.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate)

install(FILES legate/comm/coll_comm.h legate/comm/coll.h legate/comm/communicator.h
              legate/comm/communicator.inl
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/comm)

install(FILES legate/cuda/cuda.h legate/cuda/stream_pool.h legate/cuda/stream_pool.inl
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/cuda)

install(FILES legate/data/allocator.h
              legate/data/buffer.h
              legate/data/buffer.inl
              legate/data/external_allocation.h
              legate/data/external_allocation.inl
              legate/data/inline_allocation.h
              legate/data/logical_array.h
              legate/data/logical_store.h
              legate/data/physical_array.h
              legate/data/physical_array.inl
              legate/data/physical_store.h
              legate/data/physical_store.inl
              legate/data/scalar.h
              legate/data/scalar.inl
              legate/data/shape.h
              legate/data/shape.inl
              legate/data/slice.h
              legate/data/slice.inl
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/data)

install(FILES legate/experimental/trace.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/experimental)

install(FILES legate/mapping/array.h
              legate/mapping/array.inl
              legate/mapping/machine.h
              legate/mapping/machine.inl
              legate/mapping/mapping.h
              legate/mapping/mapping.inl
              legate/mapping/operation.h
              legate/mapping/operation.inl
              legate/mapping/store.h
              legate/mapping/store.inl
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/mapping)

install(FILES legate/operation/projection.h legate/operation/projection.inl
              legate/operation/task.h legate/operation/task.inl
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/operation)

install(FILES legate/partitioning/constraint.h legate/partitioning/constraint.inl
              legate/partitioning/proxy.h legate/partitioning/proxy.inl
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/partitioning)

install(FILES legate/runtime/exception_mode.h legate/runtime/library.h
              legate/runtime/library.inl legate/runtime/resource.h
              legate/runtime/runtime.h legate/runtime/runtime.inl
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/runtime)

install(FILES legate/task/exception.h
              legate/task/exception.inl
              legate/task/registrar.h
              legate/task/task.h
              legate/task/task.inl
              legate/task/task_context.h
              legate/task/task_context.inl
              legate/task/task_info.h
              legate/task/task_info.inl
              legate/task/variant_helper.h
              legate/task/variant_options.h
              legate/task/variant_options.inl
              legate/task/variant_info.h
              legate/task/variant_info.inl
              legate/task/task_signature.h
              legate/task/task_signature.inl
              legate/task/task_config.h
              legate/task/task_config.inl
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/task)

install(FILES legate/tuning/parallel_policy.h legate/tuning/parallel_policy.inl
              legate/tuning/scope.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/tuning)

install(FILES legate/type/types.h legate/type/types.inl legate/type/type_traits.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/type)

install(FILES legate/utilities/dispatch.h
              legate/utilities/hash.h
              legate/utilities/machine.h
              legate/utilities/memory.h
              legate/utilities/memory.inl
              legate/utilities/span.h
              legate/utilities/span.inl
              legate/utilities/mdspan.h
              legate/utilities/mdspan.inl
              legate/utilities/tuple.h
              legate/utilities/tuple.inl
              legate/utilities/typedefs.h
              legate/utilities/shared_ptr.h
              legate/utilities/shared_ptr.inl
              legate/utilities/internal_shared_ptr.h
              legate/utilities/internal_shared_ptr.inl
              legate/utilities/cpp_version.h
              legate/utilities/assert.h
              legate/utilities/abort.h
              legate/utilities/scope_guard.h
              legate/utilities/scope_guard.inl
              legate/utilities/compiler.h
              legate/utilities/macros.h
              legate/utilities/proc_local_storage.h
              legate/utilities/proc_local_storage.inl
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/utilities)

install(FILES legate/utilities/detail/compressed_pair.h
              legate/utilities/detail/shared_ptr_control_block.h
              legate/utilities/detail/shared_ptr_control_block.inl
              legate/utilities/detail/type_traits.h
              legate/utilities/detail/zip.h
              legate/utilities/detail/zip.inl
              legate/utilities/detail/enumerate.h
              legate/utilities/detail/enumerate.inl
              legate/utilities/detail/zstring_view.h
              legate/utilities/detail/zstring_view.inl
              legate/utilities/detail/doxygen.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/utilities/detail)

install(FILES legate/utilities/detail/mdspan/flat_mdspan_view.h
              legate/utilities/detail/mdspan/flat_mdspan_view.inl
              legate/utilities/detail/mdspan/flat_mdspan_iterator.h
              legate/utilities/detail/mdspan/flat_mdspan_iterator.inl
              legate/utilities/detail/mdspan/reduction_accessor.h
              legate/utilities/detail/mdspan/reduction_accessor.inl
              legate/utilities/detail/mdspan/for_each_in_extent.h
              legate/utilities/detail/mdspan/for_each_in_extent.inl
              legate/utilities/detail/mdspan/util.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/utilities/detail/mdspan)

install(FILES legate/utilities/detail/dlpack/dlpack.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/utilities/detail/dlpack)

# ########################################################################################
# * install Legate I/O -----------------------------------------------------------

install(FILES legate/io/hdf5/interface.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/io/hdf5)

# ########################################################################################
# * install Legate STL -----------------------------------------------------------

install(FILES legate/experimental/stl.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/experimental)

install(FILES legate/experimental/stl/detail/for_each.hpp
              legate/experimental/stl/detail/span.hpp
              legate/experimental/stl/detail/registrar.hpp
              legate/experimental/stl/detail/transform_reduce.hpp
              legate/experimental/stl/detail/stlfwd.hpp
              legate/experimental/stl/detail/get_logical_store.hpp
              legate/experimental/stl/detail/config.hpp
              legate/experimental/stl/detail/elementwise.hpp
              legate/experimental/stl/detail/functional.hpp
              legate/experimental/stl/detail/meta.hpp
              legate/experimental/stl/detail/mdspan.hpp
              legate/experimental/stl/detail/suffix.hpp
              legate/experimental/stl/detail/prefix.hpp
              legate/experimental/stl/detail/slice.hpp
              legate/experimental/stl/detail/type_traits.hpp
              legate/experimental/stl/detail/transform.hpp
              legate/experimental/stl/detail/iterator.hpp
              legate/experimental/stl/detail/utility.hpp
              legate/experimental/stl/detail/store.hpp
              legate/experimental/stl/detail/launch_task.hpp
              legate/experimental/stl/detail/ranges.hpp
              legate/experimental/stl/detail/reduce.hpp
              legate/experimental/stl/detail/fill.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/experimental/stl/detail)

# ########################################################################################
# * install Legate timing -----------------------------------------------------------

install(FILES legate/timing/timing.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/timing)

# ########################################################################################
# * install Legate MPI Wrapper -----------------------------------------------------------
#
# We want to install the entire CMake project for the wrapper as-is into the share folder
# in the install path. That way -- once installed -- the user can build the wrapper
# themselves. This is also why we specifically avoided generating the install rules
install(FILES ${LEGATE_DIR}/share/legate/mpi_wrapper/install.bash
              ${LEGATE_DIR}/share/legate/mpi_wrapper/CMakeLists.txt
        DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/legate/mpi_wrapper")

install(FILES ${LEGATE_DIR}/share/legate/mpi_wrapper/cmake/Config.cmake.in
        DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/legate/mpi_wrapper/cmake")

install(FILES ${LEGATE_DIR}/share/legate/mpi_wrapper/src/legate_mpi_wrapper/mpi_wrapper.c
              ${LEGATE_DIR}/share/legate/mpi_wrapper/src/legate_mpi_wrapper/mpi_wrapper.h
              ${LEGATE_DIR}/share/legate/mpi_wrapper/src/legate_mpi_wrapper/mpi_wrapper_types.h
        DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/legate/mpi_wrapper/legate_mpi_wrapper")

# ########################################################################################
# * install Realm MPI Bootstrap ----------------------------------------------------------
#
# We want to install the entire CMake project for the bootstrap as-is into the share
# folder in the install path. That way -- once installed -- the user can build the
# bootstrap themselves. This is also why we specifically avoided generating the install
# rules
install(FILES ${LEGATE_DIR}/share/legate/realm_ucp_bootstrap/CMakeLists.txt
              ${LEGATE_DIR}/share/legate/realm_ucp_bootstrap/bootstrap.h
              ${LEGATE_DIR}/share/legate/realm_ucp_bootstrap/bootstrap_mpi.c
              ${LEGATE_DIR}/share/legate/realm_ucp_bootstrap/bootstrap_util.h
        DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/legate/realm_ucp_bootstrap")

install(FILES ${LEGATE_DIR}/share/legate/realm_ucp_bootstrap/cmake/Config.cmake.in
        DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/legate/realm_ucp_bootstrap/cmake")

install(FILES ${LEGATE_DIR}/share/legate/sanitizers/asan_default_options.txt
              ${LEGATE_DIR}/share/legate/sanitizers/lsan_suppressions.txt
              ${LEGATE_DIR}/share/legate/sanitizers/tsan_default_options.txt
              ${LEGATE_DIR}/share/legate/sanitizers/tsan_suppressions.txt
              ${LEGATE_DIR}/share/legate/sanitizers/ubsan_default_options.txt
              ${LEGATE_DIR}/share/legate/sanitizers/ubsan_suppressions.txt
        DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/legate/sanitizers")

# ########################################################################################
# * install Legate examples -----------------------------------------------------------

install(FILES ${LEGATE_DIR}/share/legate/examples/helloworld/helloworld.cc
              ${LEGATE_DIR}/share/legate/examples/helloworld/CMakeLists.txt
              ${LEGATE_DIR}/share/legate/examples/helloworld/build-and-run.sh
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/legate/examples/helloworld)
legate_add_external_tidy_target(ROOT_DIR ${LEGATE_DIR}/share/legate/examples/helloworld
                                SOURCES helloworld.cc)

install(FILES ${LEGATE_DIR}/share/legate/examples/saxpy/saxpy.cc
              ${LEGATE_DIR}/share/legate/examples/saxpy/CMakeLists.txt
              ${LEGATE_DIR}/share/legate/examples/saxpy/build-and-run.sh
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/legate/examples/saxpy)
legate_add_external_tidy_target(ROOT_DIR ${LEGATE_DIR}/share/legate/examples/saxpy
                                SOURCES saxpy.cc)

install(FILES ${LEGATE_DIR}/share/legate/examples/manual_tasks/manual_tasks.cc
              ${LEGATE_DIR}/share/legate/examples/manual_tasks/CMakeLists.txt
              ${LEGATE_DIR}/share/legate/examples/manual_tasks/build-and-run.sh
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/legate/examples/manual_tasks)
legate_add_external_tidy_target(ROOT_DIR ${LEGATE_DIR}/share/legate/examples/manual_tasks
                                SOURCES manual_tasks.cc)

install(FILES ${LEGATE_DIR}/share/legate/examples/io/hdf5/ex1.py
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/legate/examples/io/hdf5)

install(FILES ${LEGATE_DIR}/share/legate/examples/io/hdf5/ex1_gen/gen_h5_data.c
              ${LEGATE_DIR}/share/legate/examples/io/hdf5/ex1_gen/CMakeLists.txt
              ${LEGATE_DIR}/share/legate/examples/io/hdf5/ex1_gen/build.bash
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/legate/examples/io/hdf5/ex1_gen)

install(FILES ${LEGATE_DIR}/share/legate/examples/binding/manual/CMakeLists.txt
              ${LEGATE_DIR}/share/legate/examples/binding/manual/hello_world.cc
              ${LEGATE_DIR}/share/legate/examples/binding/manual/hello_world.py
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/legate/examples/binding/manual)
legate_add_external_tidy_target(ROOT_DIR ${LEGATE_DIR}/share/legate/examples/binding/manual
                                SOURCES hello_world.cc)

# Not adding these files as tidy target, because it requires calling pip install to
# configure.
install(FILES ${LEGATE_DIR}/share/legate/examples/binding/cython/hello_world.cc
              ${LEGATE_DIR}/share/legate/examples/binding/cython/hello_world.h
              ${LEGATE_DIR}/share/legate/examples/binding/cython/hello_world_cython.pyx
              ${LEGATE_DIR}/share/legate/examples/binding/cython/hello_world.py
              ${LEGATE_DIR}/share/legate/examples/binding/cython/pyproject.toml
              ${LEGATE_DIR}/share/legate/examples/binding/cython/CMakeLists.txt
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/legate/examples/binding/cython)

install(FILES ${LEGATE_DIR}/share/legate/examples/binding/pybind11/hello_world.cc
              ${LEGATE_DIR}/share/legate/examples/binding/pybind11/hello_world.py
              ${LEGATE_DIR}/share/legate/examples/binding/pybind11/pyproject.toml
              ${LEGATE_DIR}/share/legate/examples/binding/pybind11/CMakeLists.txt
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/legate/examples/binding/pybind11)
legate_add_external_tidy_target(ROOT_DIR ${LEGATE_DIR}/share/legate/examples/binding/pybind11
                                SOURCES hello_world.cc)

# ########################################################################################
# * install legate-bind.sh  --------------------------------------------------------

# This is non-standard. Normally this should just go under a top-level /libexec directory
# (or possible /lib/legate/libexec), but we must put it under /share because of
# restrictions placed on us by scikit-build-core. scikit-build-core only considers a
# select subset of directories under $PREFIX when collecting files to populate a wheel and
# while /share gets picked up, /libexec does *not*.
#
# It's possible this restriction is relaxed in the future, but for now, /share it must be.
install(PROGRAMS ${LEGATE_DIR}/share/legate/libexec/legate-bind.sh
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/legate/${CMAKE_INSTALL_LIBEXECDIR})

# ########################################################################################
# * install Legate CMake Helpers ---------------------------------------------------------

install(FILES ${LEGATE_CMAKE_DIR}/Modules/legate_configure_target.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/legate/Modules)

# ########################################################################################
# * install Legate debug symbols ---------------------------------------------------------

legate_install_debug_symbols(TARGET legate INSTALL_DIR ${CMAKE_INSTALL_LIBDIR} RECURSIVE)

# ########################################################################################
# * install export -----------------------------------------------------------

set(doc_string
    [=[
Provide targets for Legate, the Foundation for All Legate Libraries.

Imported Targets:
  - legate::legate

]=])

# Normally this is done transparently (via the "code_string" below, embedded in the
# Findlegate.cmake) if the CMakeLists.txt calling this one finds the legate via a
# find_package() call. But if we are being built as a subdirectory, then we need to
# explicitly set(<the_variable> ... PARENT_SCOPE) in order for downstream to see it...
if(NOT PROJECT_IS_TOP_LEVEL)
  # These must match the decls below BEGIN MUST MATCH
  set_parent_scope(Legion_USE_CUDA)
  set_parent_scope(legate_USE_CUDA)
  set_parent_scope(Legion_USE_OpenMP)
  set_parent_scope(Legion_USE_Python)
  set_parent_scope(Legion_CUDA_ARCH)
  set_parent_scope(Legion_NETWORKS)
  set_parent_scope(Legion_BOUNDS_CHECKS)
  set_parent_scope(Legion_MAX_DIM)
  set_parent_scope(Legion_MAX_FIELDS)
  # END MUST MATCH
endif()

string(JOIN
       "\n"
       base_code_string
       [=[
if(NOT TARGET CCCL::Thrust)
  thrust_create_target(CCCL::Thrust FROM_OPTIONS)
endif()
]=]
       # These must match the decls above BEGIN MUST MATCH
       "set(Legion_USE_CUDA ${Legion_USE_CUDA})"
       "set(legate_USE_CUDA ${legate_USE_CUDA})"
       "set(Legion_USE_OpenMP ${Legion_USE_OpenMP})"
       "set(Legion_USE_Python ${Legion_USE_Python})"
       "set(Legion_CUDA_ARCH ${Legion_CUDA_ARCH})"
       "set(Legion_NETWORKS ${Legion_NETWORKS})"
       "set(Legion_BOUNDS_CHECKS ${Legion_BOUNDS_CHECKS})"
       "set(Legion_MAX_DIM ${Legion_MAX_DIM})"
       "set(Legion_MAX_FIELDS ${Legion_MAX_FIELDS})"
       # END MUST MATCH
)

get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)

list(REMOVE_ITEM languages NONE)

message(STATUS "Enabled languages: ${languages}")

string(JOIN "\n" install_code_string "${base_code_string}" [=[
include("${CMAKE_CURRENT_LIST_DIR}/Modules/legate_configure_target.cmake")
]=])
# FIXME(wonchanl): Passing LANGUAGES triggers a bug in rapids-cmake. Put it back once we
# bump the rapids-cmake version.
rapids_export(INSTALL legate
              EXPORT_SET legate-exports
              GLOBAL_TARGETS legate
              NAMESPACE legate::
              DOCUMENTATION doc_string
              FINAL_CODE_BLOCK install_code_string)

string(JOIN "\n" build_code_string "${base_code_string}"
       "include(\"${LEGATE_CMAKE_DIR}/Modules/legate_configure_target.cmake\")")

# build export targets
# FIXME(wonchanl): Passing LANGUAGES triggers a bug in rapids-cmake. Put it back once we
# bump the rapids-cmake version.
rapids_export(BUILD legate
              EXPORT_SET legate-exports
              GLOBAL_TARGETS legate
              NAMESPACE legate::
              DOCUMENTATION doc_string
              FINAL_CODE_BLOCK build_code_string)

set(legate_ROOT ${CMAKE_CURRENT_BINARY_DIR})

if(legate_BUILD_TESTS)
  include(CTest)

  add_subdirectory(${LEGATE_DIR}/tests/cpp tests)
endif()

if(legate_BUILD_BENCHMARKS)
  add_subdirectory(${LEGATE_DIR}/benchmarks/cpp benchmarks)
endif()

include("${LEGATE_CMAKE_DIR}/Modules/generate_install_info.cmake")

legate_generate_install_info_py()

if(PROJECT_IS_TOP_LEVEL)
  include("${LEGATE_CMAKE_DIR}/Modules/uninstall.cmake")

  legate_uninstall_target(TARGET uninstall)
endif()

# This call must come LAST because cpm_licenses_create_disclaimer_target only works
# properly if it is called after all third-party packages have been downloaded.  Do not
# move this call!
set(legate_LICENSE_FILE "${CMAKE_CURRENT_BINARY_DIR}/licenses.txt")
cpm_licenses_create_disclaimer_target(legate_write_licenses "${legate_LICENSE_FILE}"
                                      "${CPM_PACKAGES}")

install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target legate_write_licenses
                COMMAND_ERROR_IS_FATAL ANY)")
install(FILES ${legate_LICENSE_FILE} DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/legate")

if(legate_BUILD_DOCS)
  add_subdirectory(${LEGATE_DIR}/docs/legate docs/legate)
endif()

include("${LEGATE_CMAKE_DIR}/Modules/export_aedifix_post_config.cmake")

legate_export_aedifix_post_config()

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
