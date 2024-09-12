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

list(APPEND CMAKE_MESSAGE_CONTEXT "cpp")

# ########################################################################################
# * User Options  ------------------------------------------------------------

include(cmake/Modules/legate_options.cmake)

# ########################################################################################
# * Project definition -------------------------------------------------------

include(GNUInstallDirs)

# Write the version header
rapids_cmake_write_version_file(${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/version.h
                                PREFIX LEGATE)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/compile_commands.cmake)

legate_export_compile_commands()

# ########################################################################################
# * Build Type ---------------------------------------------------------------

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# ########################################################################################
# * conda environment --------------------------------------------------------

rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)

# We're building python extension libraries, which must always be installed under lib/,
# even if the system normally uses lib64/. Rapids-cmake currently doesn't realize this
# when we're going through scikit-build, see
# https://github.com/rapidsai/rapids-cmake/issues/426 Do this before we include Legion, so
# its build also inherits this setting.
if(TARGET conda_env)
  set(CMAKE_INSTALL_LIBDIR "lib")
endif()

# ########################################################################################
# * Dependencies -------------------------------------------------------------

# add third party dependencies using CPM
set(legate_VERSIONS_JSON "${LEGATE_DIR}/cmake/versions/versions.json")
rapids_cpm_init(OVERRIDE ${legate_VERSIONS_JSON})

include(${LEGATE_DIR}/cmake/Modules/find_or_configure.cmake)

# ########################################################################################
# * Default Flags -------------------------------------------------------------

include(${LEGATE_DIR}/cmake/Modules/default_flags.cmake)

legate_configure_default_compiler_flags()
legate_configure_default_linker_flags()

# ########################################################################################
# * CCCL ---------------------------------------------------------------------

# Pull this in before Legion, so that Legion will use the same libcu++ as Legate (the one
# pull from CCCL)

legate_find_or_configure(PACKAGE cccl)

# ########################################################################################
# * Python -------------------------------------------------------------------

macro(_find_package_python3)
  rapids_find_package(Python3
                      BUILD_EXPORT_SET legate-exports
                      INSTALL_EXPORT_SET legate-exports
                      COMPONENTS Interpreter Development
                      FIND_ARGS
                      REQUIRED)
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
  _find_package_python3()
  if(Python3_FOUND AND Python3_VERSION)
    set(Legion_Python_Version ${Python3_VERSION})
  endif()
endif()

include(${LEGATE_DIR}/cmake/Modules/cuda_arch_helpers.cmake)

if(Legion_USE_CUDA)
  # Needs to run before find_package(Legion)
  set_cuda_arch_from_names()
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
  _find_package_python3()
endif()

if(Legion_USE_CUDA)
  # Enable the CUDA language
  enable_language(CUDA)
  # Must come after `enable_language(CUDA)` Use `-isystem <path>` instead of
  # `-isystem=<path>` because the former works with clangd intellisense
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
  # Find NCCL
  legate_find_or_configure(PACKAGE NCCL)
endif()

# ########################################################################################
# * MPI wrapper --------------------------------------------------------------

if(Legion_NETWORKS)
  # The wrapper we build should not install anything but the library object itself,
  # because we want to install the sources (and cmake files) to a different location
  # ourselves.
  set(LEGATE_MPI_WRAPPER_SRC_INSTALL_RULES OFF)

  add_subdirectory(share/legate/mpi_wrapper)
endif()

# ########################################################################################
# * std::span ----------------------------------------------------------------

legate_find_or_configure(PACKAGE span)

# ########################################################################################
# * std::mdspan --------------------------------------------------------------

legate_find_or_configure(PACKAGE mdspan)

# ########################################################################################
# * fmt::fmt --------------------------------------------------------------

legate_find_or_configure(PACKAGE fmt)

# ########################################################################################
# * argparse::argparse --------------------------------------------------------------

legate_find_or_configure(PACKAGE argparse)

# ########################################################################################
# * legate --------------------------------------------------------------

include(cmake/Modules/generate_sanitizer_options.cmake)

legate_generate_sanitizer_options(
  SRC share/legate/sanitizers/asan_default_options.txt
  DELIM [[:]]
  DEST "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/asan_default_options.h"
)

legate_generate_sanitizer_options(
  SRC share/legate/sanitizers/lsan_suppressions.txt
  DELIM [[\n]]
  DEST "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/lsan_suppressions.h"
)

legate_generate_sanitizer_options(
  SRC share/legate/sanitizers/ubsan_default_options.txt
  DELIM [[:]]
  DEST "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/ubsan_default_options.h"
)

legate_generate_sanitizer_options(
  SRC share/legate/sanitizers/tsan_suppressions.txt
  DELIM [[\n]]
  DEST "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/tsan_suppressions.h"
)
legate_generate_sanitizer_options(
  SRC share/legate/sanitizers/tsan_default_options.txt
  DELIM [[:]]
  DEST "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/tsan_default_options.h"
)

list(APPEND
     legate_SOURCES
     src/legate/comm/coll.cc
     src/legate/comm/detail/backend_network.cc
     src/legate/comm/detail/comm.cc
     src/legate/comm/detail/comm_cpu.cc
     src/legate/comm/detail/comm_local.cc
     src/legate/comm/detail/local_network.cc
     src/legate/comm/detail/logger.cc
     src/legate/comm/detail/thread_comm.cc
     src/legate/cuda/stream_pool.cc
     src/legate/data/allocator.cc
     src/legate/data/external_allocation.cc
     src/legate/data/logical_array.cc
     src/legate/data/logical_store.cc
     src/legate/data/scalar.cc
     src/legate/data/shape.cc
     src/legate/data/physical_array.cc
     src/legate/data/physical_store.cc
     src/legate/data/detail/array_tasks.cc
     src/legate/data/detail/attachment.cc
     src/legate/data/detail/external_allocation.cc
     src/legate/data/detail/logical_array.cc
     src/legate/data/detail/logical_region_field.cc
     src/legate/data/detail/logical_store.cc
     src/legate/data/detail/scalar.cc
     src/legate/data/detail/physical_array.cc
     src/legate/data/detail/physical_store.cc
     src/legate/data/detail/shape.cc
     src/legate/data/detail/transform.cc
     src/legate/data/detail/future_wrapper.cc
     src/legate/data/detail/region_field.cc
     src/legate/experimental/trace.cc
     src/legate/mapping/array.cc
     src/legate/mapping/machine.cc
     src/legate/mapping/mapping.cc
     src/legate/mapping/operation.cc
     src/legate/mapping/store.cc
     src/legate/mapping/detail/array.cc
     src/legate/mapping/detail/base_mapper.cc
     src/legate/mapping/detail/core_mapper.cc
     src/legate/mapping/detail/instance_manager.cc
     src/legate/mapping/detail/machine.cc
     src/legate/mapping/detail/mapping.cc
     src/legate/mapping/detail/operation.cc
     src/legate/mapping/detail/store.cc
     src/legate/operation/projection.cc
     src/legate/operation/task.cc
     src/legate/operation/detail/attach.cc
     src/legate/operation/detail/copy.cc
     src/legate/operation/detail/copy_launcher.cc
     src/legate/operation/detail/discard.cc
     src/legate/operation/detail/execution_fence.cc
     src/legate/operation/detail/fill.cc
     src/legate/operation/detail/fill_launcher.cc
     src/legate/operation/detail/gather.cc
     src/legate/operation/detail/index_attach.cc
     src/legate/operation/detail/mapping_fence.cc
     src/legate/operation/detail/launcher_arg.cc
     src/legate/operation/detail/operation.cc
     src/legate/operation/detail/store_projection.cc
     src/legate/operation/detail/reduce.cc
     src/legate/operation/detail/req_analyzer.cc
     src/legate/operation/detail/scatter.cc
     src/legate/operation/detail/scatter_gather.cc
     src/legate/operation/detail/task.cc
     src/legate/operation/detail/task_launcher.cc
     src/legate/operation/detail/timing.cc
     src/legate/operation/detail/unmap_and_detach.cc
     src/legate/partitioning/constraint.cc
     src/legate/partitioning/detail/constraint.cc
     src/legate/partitioning/detail/constraint_solver.cc
     src/legate/partitioning/detail/partition.cc
     src/legate/partitioning/detail/partitioner.cc
     src/legate/partitioning/detail/partitioning_tasks.cc
     src/legate/partitioning/detail/restriction.cc
     src/legate/runtime/library.cc
     src/legate/runtime/runtime.cc
     src/legate/runtime/scope.cc
     src/legate/runtime/detail/communicator_manager.cc
     src/legate/runtime/detail/field_manager.cc
     src/legate/runtime/detail/library.cc
     src/legate/runtime/detail/partition_manager.cc
     src/legate/runtime/detail/projection.cc
     src/legate/runtime/detail/region_manager.cc
     src/legate/runtime/detail/runtime.cc
     src/legate/runtime/detail/shard.cc
     src/legate/runtime/detail/config.cc
     src/legate/runtime/detail/mapper_manager.cc
     src/legate/runtime/detail/argument_parsing.cc
     src/legate/task/registrar.cc
     src/legate/task/task.cc
     src/legate/task/task_context.cc
     src/legate/task/task_info.cc
     src/legate/task/variant_options.cc
     src/legate/task/detail/return_value.cc
     src/legate/task/detail/returned_exception.cc
     src/legate/task/detail/returned_cpp_exception.cc
     src/legate/task/detail/returned_python_exception.cc
     src/legate/task/detail/task_context.cc
     src/legate/task/detail/inline_task_body.cc
     src/legate/task/detail/legion_task_body.cc
     src/legate/task/detail/task.cc
     src/legate/task/detail/task_return.cc
     src/legate/task/detail/task_return_layout.cc
     src/legate/type/type_info.cc
     src/legate/type/detail/type_info.cc
     src/legate/utilities/debug.cc
     src/legate/utilities/machine.cc
     src/legate/utilities/linearize.cc
     src/legate/utilities/internal_shared_ptr.cc
     src/legate/utilities/compiler.cc
     src/legate/utilities/abort.cc
     src/legate/utilities/detail/buffer_builder.cc
     src/legate/utilities/detail/env.cc
     src/legate/utilities/detail/tuple.cc
     src/legate/utilities/detail/deserializer.cc
     src/legate/utilities/detail/formatters.cc
     src/legate/timing/timing.cc
     # stl
     src/legate/experimental/stl/detail/clang_tidy_dummy.cpp)

if(legate_ENABLE_SANITIZERS)
  list(APPEND legate_SOURCES src/legate/utilities/detail/sanitizer_defaults.cc)
endif()

if(Legion_NETWORKS)
  list(APPEND legate_SOURCES src/legate/comm/detail/mpi_network.cc
       src/legate/comm/detail/mpi_interface.cc src/legate/comm/detail/comm_mpi.cc)
endif()

if(Legion_USE_OpenMP)
  list(APPEND legate_SOURCES src/legate/data/detail/array_tasks_omp.cc
       src/legate/partitioning/detail/partitioning_tasks_omp.cc)
endif()

if(Legion_USE_CUDA)
  list(APPEND legate_SOURCES src/legate/comm/detail/comm_nccl.cu
       src/legate/data/detail/array_tasks.cu
       src/legate/partitioning/detail/partitioning_tasks.cu)
  if(CAL_DIR)
    list(APPEND legate_SOURCES src/legate/comm/detail/comm_cal.cu)
  endif()
endif()

add_library(legate ${legate_SOURCES})
add_library(legate::legate ALIAS legate)

set(legate_CXX_PRIVATE_OPTIONS "")
set(legate_CUDA_PRIVATE_OPTIONS "")
set(legate_CXX_PUBLIC_OPTIONS "")
set(legate_CUDA_PUBLIC_OPTIONS "")
set(legate_LINKER_OPTIONS "")

include(${LEGATE_DIR}/cmake/Modules/set_cpu_arch_flags.cmake)

set_cpu_arch_flags(legate_CXX_PRIVATE_OPTIONS)

if(Legion_USE_CUDA)
  add_cuda_architecture_defines(legate_CUDA_PUBLIC_OPTIONS ARCHS ${Legion_CUDA_ARCH})

  list(APPEND legate_CUDA_PRIVATE_OPTIONS -Xfatbin=-compress-all)
  list(APPEND legate_CUDA_PRIVATE_OPTIONS --expt-extended-lambda)
  list(APPEND legate_CUDA_PRIVATE_OPTIONS --expt-relaxed-constexpr)
  list(APPEND legate_CUDA_PRIVATE_OPTIONS -Wno-deprecated-gpu-targets)
endif()

# Change THRUST_DEVICE_SYSTEM for `.cpp` files If we include Thrust in "CUDA mode" in .cc
# files, that ends up pulling the definition of __half from the CUDA toolkit, and Legion
# defines a custom __half when compiling outside of nvcc (because CUDA's __half doesn't
# define any __host__ functions), which causes a conflict.
if(Legion_USE_OpenMP)
  rapids_find_package(OpenMP
                      GLOBAL_TARGETS OpenMP::OpenMP_CXX
                      BUILD_EXPORT_SET legate-exports
                      INSTALL_EXPORT_SET legate-exports
                      COMPONENTS CXX
                      FIND_ARGS
                      REQUIRED)

  list(APPEND legate_CXX_PUBLIC_OPTIONS -UTHRUST_DEVICE_SYSTEM)
  list(APPEND legate_CXX_PUBLIC_OPTIONS -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP)
else()
  list(APPEND legate_CXX_PUBLIC_OPTIONS -UTHRUST_DEVICE_SYSTEM)
  list(APPEND legate_CXX_PUBLIC_OPTIONS -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(LEGATE_PLATFORM_RPATH_ORIGIN "\$ORIGIN")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(LEGATE_PLATFORM_RPATH_ORIGIN "@loader_path")
else()
  message(FATAL_ERROR "Unsupported system: ${CMAKE_SYSTEM_NAME}, don't know how to set rpath 'origin' on this platform"
  )
endif()

set_target_properties(legate
                      PROPERTIES EXPORT_NAME legate
                                 LIBRARY_OUTPUT_NAME legate
                                 BUILD_RPATH "${LEGATE_PLATFORM_RPATH_ORIGIN}"
                                 INSTALL_RPATH "${LEGATE_PLATFORM_RPATH_ORIGIN}"
                                 CXX_STANDARD ${CMAKE_CXX_STANDARD}
                                 CXX_STANDARD_REQUIRED ON
                                 CUDA_STANDARD ${CMAKE_CUDA_STANDARD}
                                 CUDA_STANDARD_REQUIRED ON
                                 POSITION_INDEPENDENT_CODE ON
                                 INTERFACE_POSITION_INDEPENDENT_CODE ON
                                 LIBRARY_OUTPUT_DIRECTORY lib
                                 SOVERSION ${legate_version})

# export this so that install_info.py can properly locate the versioned and unversioned
# library names
set_property(TARGET legate APPEND PROPERTY EXPORT_PROPERTIES LIBRARY_OUTPUT_NAME)

if(Legion_USE_CUDA)
  set_property(TARGET legate PROPERTY CUDA_ARCHITECTURES ${Legion_CUDA_ARCH})
endif()

target_link_libraries(legate
                      # Order is important here. We want conda includes to go last because
                      # the conda env might contain other versions of the libraries below
                      # (for example CCCL). We want the conda includes to go last so that
                      # we ensure that compiler pick up the right headers.
                      #
                      # This is also why CCCL::Thrust comes *before* Legion, because
                      # Legion may pick up headers from inside the conda environment.
                      #
                      # Similarly, we want the MPI wrapper to load as early as possible,
                      # since it provides our MPI symbols.
                      PUBLIC $<TARGET_NAME_IF_EXISTS:legate::mpi_wrapper>
                             CCCL::Thrust
                             Legion::Legion
                             # See
                             # https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html#nvtx3
                             $<TARGET_NAME_IF_EXISTS:$<IF:$<TARGET_EXISTS:CUDA::nvtx3>,CUDA::nvtx3,CUDA::nvToolsExt>>
                             $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
                             $<TARGET_NAME_IF_EXISTS:std::mdspan>
                             $<TARGET_NAME_IF_EXISTS:std::span>
                      PRIVATE $<TARGET_NAME_IF_EXISTS:NCCL::NCCL> fmt::fmt
                              argparse::argparse $<TARGET_NAME_IF_EXISTS:conda_env>)

if(Legion_USE_CUDA)
  if(legate_STATIC_CUDA_RUNTIME)
    set_target_properties(legate PROPERTIES CUDA_RUNTIME_LIBRARY Static)
    # Make sure to export to consumers what runtime we used
    target_link_libraries(legate PUBLIC CUDA::cudart_static)
  else()
    set_target_properties(legate PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
    # Make sure to export to consumers what runtime we used
    target_link_libraries(legate PUBLIC CUDA::cudart)
  endif()
endif()

if(Legion_USE_CUDA AND CAL_DIR)
  message(VERBOSE "legate: CAL_DIR ${CAL_DIR}")
  target_include_directories(legate PRIVATE ${CAL_DIR}/include)
  target_link_libraries(legate PRIVATE ${CAL_DIR}/lib/libcal.so)
endif()

# ########################################################################################
# * Custom User Flags --------------------------------------------------------

macro(legate_add_target_compile_option TARGET_NAME OPTION_LANG VIS OPTION_NAME)
  if(NOT ("${${OPTION_NAME}}" MATCHES ".*;.*"))
    # Using this form of separate_arguments() makes sure that quotes are respected when
    # the list is formed. Otherwise stuff like
    #
    # "--compiler-options='-foo -bar -baz'"
    #
    # becomes
    #
    # --compiler-options="'-foo";"-bar";"-baz'"
    #
    # which is obviously not what we wanted
    separate_arguments(${OPTION_NAME} NATIVE_COMMAND "${${OPTION_NAME}}")
  endif()
  if(${OPTION_NAME})
    target_compile_options(${TARGET_NAME} ${VIS}
                           "$<$<COMPILE_LANGUAGE:${OPTION_LANG}>:${${OPTION_NAME}}>")
  endif()
endmacro()

macro(legate_add_target_link_option TARGET_NAME VIS OPTION_NAME)
  if(NOT ("${${OPTION_NAME}}" MATCHES ".*;.*"))
    separate_arguments(${OPTION_NAME} NATIVE_COMMAND "${${OPTION_NAME}}")
  endif()
  if(${OPTION_NAME})
    target_link_options(${TARGET_NAME} ${VIS} "${${OPTION_NAME}}")
  endif()
endmacro()

function(check_nvcc_pedantic_flags)
  if(legate_SKIP_NVCC_PEDANTIC_CHECK)
    message(VERBOSE "Skipping nvcc pedantic check (explicitly skipped by user)")
    return()
  endif()
  if(NOT (CMAKE_CUDA_COMPILER_ID MATCHES "NVIDIA"))
    message(VERBOSE
            "Skipping nvcc pedantic check (compiler \"${CMAKE_CUDA_COMPILER_ID}\" is not nvcc)"
    )
    return()
  endif()
  # We want to catch either "-pedantic" or "--compiler-option=-pedantic" or
  # --compiler-options='-pedantic' but we do NOT want to catch -Wformat-pedantic!
  string(REGEX MATCH [=[[ |=|='|="]\-W?pedantic]=] match_var "${legate_CUDA_FLAGS}")
  if(match_var)
    message(FATAL_ERROR "-pedantic (or -Wpedantic) is not supported by nvcc and will lead to "
                        "spurious warnings in generated code. Please remove it from your build flags. If "
                        "you would like to override this behavior, reconfigure with "
                        "-Dlegate_SKIP_NVCC_PEDANTIC_CHECK=ON.")
  endif()
endfunction()
check_nvcc_pedantic_flags()

legate_add_target_compile_option(legate CXX PRIVATE legate_CXX_PRIVATE_OPTIONS)
legate_add_target_compile_option(legate CUDA PRIVATE legate_CUDA_PRIVATE_OPTIONS)

legate_add_target_compile_option(legate CXX PUBLIC legate_CXX_PUBLIC_OPTIONS)
legate_add_target_compile_option(legate CUDA PUBLIC legate_CUDA_PUBLIC_OPTIONS)

legate_add_target_compile_option(legate CXX PRIVATE legate_CXX_FLAGS)
legate_add_target_compile_option(legate CUDA PRIVATE legate_CUDA_FLAGS)

legate_add_target_link_option(legate PUBLIC legate_LINKER_FLAGS)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/generate_legate_defines.cmake)

legate_generate_legate_defines()

set(legate_LOCAL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_include_directories(legate
                           PUBLIC $<BUILD_INTERFACE:${legate_LOCAL_INCLUDE_DIR}>
                                  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/legate>
                           INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/legate>
)

if(legate_BUILD_DOCS)
  add_subdirectory(docs/legate)
endif()

# ########################################################################################
# * install targets-----------------------------------------------------------

include(CPack)

rapids_cmake_install_lib_dir(lib_dir)

install(TARGETS legate DESTINATION ${lib_dir} EXPORT legate-exports)

install(FILES src/legate.h
              ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/legate/legate_defines.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/version.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate)

install(FILES src/legate/comm/coll_comm.h src/legate/comm/coll.h
              src/legate/comm/communicator.h src/legate/comm/communicator.inl
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/comm)

install(FILES src/legate/cuda/cuda.h src/legate/cuda/stream_pool.h
              src/legate/cuda/stream_pool.inl
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/cuda)

install(FILES src/legate/data/allocator.h
              src/legate/data/buffer.h
              src/legate/data/external_allocation.h
              src/legate/data/external_allocation.inl
              src/legate/data/inline_allocation.h
              src/legate/data/logical_array.h
              src/legate/data/logical_array.inl
              src/legate/data/logical_store.h
              src/legate/data/logical_store.inl
              src/legate/data/physical_array.h
              src/legate/data/physical_array.inl
              src/legate/data/physical_store.h
              src/legate/data/physical_store.inl
              src/legate/data/scalar.h
              src/legate/data/scalar.inl
              src/legate/data/shape.h
              src/legate/data/shape.inl
              src/legate/data/slice.h
              src/legate/data/slice.inl
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/data)

install(FILES src/legate/experimental/trace.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/experimental)

install(FILES src/legate/mapping/array.h
              src/legate/mapping/array.inl
              src/legate/mapping/machine.h
              src/legate/mapping/machine.inl
              src/legate/mapping/mapping.h
              src/legate/mapping/mapping.inl
              src/legate/mapping/operation.h
              src/legate/mapping/operation.inl
              src/legate/mapping/store.h
              src/legate/mapping/store.inl
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/mapping)

install(FILES src/legate/operation/projection.h src/legate/operation/projection.inl
              src/legate/operation/task.h src/legate/operation/task.inl
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/operation)

install(FILES src/legate/partitioning/constraint.h src/legate/partitioning/constraint.inl
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/partitioning)

install(FILES src/legate/runtime/exception_mode.h
              src/legate/runtime/library.h
              src/legate/runtime/library.inl
              src/legate/runtime/resource.h
              src/legate/runtime/runtime.h
              src/legate/runtime/runtime.inl
              src/legate/runtime/scope.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/runtime)

install(FILES src/legate/task/exception.h
              src/legate/task/exception.inl
              src/legate/task/registrar.h
              src/legate/task/task.h
              src/legate/task/task.inl
              src/legate/task/task_context.h
              src/legate/task/task_context.inl
              src/legate/task/task_info.h
              src/legate/task/task_info.inl
              src/legate/task/variant_helper.h
              src/legate/task/variant_options.h
              src/legate/task/variant_options.inl
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/task)

install(FILES src/legate/type/type_info.h src/legate/type/type_info.inl
              src/legate/type/type_traits.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/type)

install(FILES src/legate/utilities/debug.h
              src/legate/utilities/debug.inl
              src/legate/utilities/dispatch.h
              src/legate/utilities/hash.h
              src/legate/utilities/machine.h
              src/legate/utilities/memory.h
              src/legate/utilities/memory.inl
              src/legate/utilities/nvtx_help.h
              src/legate/utilities/span.h
              src/legate/utilities/span.inl
              src/legate/utilities/tuple.h
              src/legate/utilities/tuple.inl
              src/legate/utilities/typedefs.h
              src/legate/utilities/shared_ptr.h
              src/legate/utilities/shared_ptr.inl
              src/legate/utilities/internal_shared_ptr.h
              src/legate/utilities/internal_shared_ptr.inl
              src/legate/utilities/cpp_version.h
              src/legate/utilities/assert.h
              src/legate/utilities/abort.h
              src/legate/utilities/scope_guard.h
              src/legate/utilities/scope_guard.inl
              src/legate/utilities/compiler.h
              src/legate/utilities/macros.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/utilities)

install(FILES src/legate/utilities/detail/compressed_pair.h
              src/legate/utilities/detail/shared_ptr_control_block.h
              src/legate/utilities/detail/shared_ptr_control_block.inl
              src/legate/utilities/detail/type_traits.h
              src/legate/utilities/detail/zip.h
              src/legate/utilities/detail/zip.inl
              src/legate/utilities/detail/enumerate.h
              src/legate/utilities/detail/enumerate.inl
              src/legate/utilities/detail/zstring_view.h
              src/legate/utilities/detail/zstring_view.inl
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/utilities/detail)

# Legate STL headers
install(FILES src/legate/experimental/stl.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/experimental)

install(FILES src/legate/experimental/stl/detail/for_each.hpp
              src/legate/experimental/stl/detail/span.hpp
              src/legate/experimental/stl/detail/registrar.hpp
              src/legate/experimental/stl/detail/transform_reduce.hpp
              src/legate/experimental/stl/detail/stlfwd.hpp
              src/legate/experimental/stl/detail/get_logical_store.hpp
              src/legate/experimental/stl/detail/config.hpp
              src/legate/experimental/stl/detail/elementwise.hpp
              src/legate/experimental/stl/detail/functional.hpp
              src/legate/experimental/stl/detail/meta.hpp
              src/legate/experimental/stl/detail/mdspan.hpp
              src/legate/experimental/stl/detail/suffix.hpp
              src/legate/experimental/stl/detail/prefix.hpp
              src/legate/experimental/stl/detail/slice.hpp
              src/legate/experimental/stl/detail/type_traits.hpp
              src/legate/experimental/stl/detail/transform.hpp
              src/legate/experimental/stl/detail/iterator.hpp
              src/legate/experimental/stl/detail/utility.hpp
              src/legate/experimental/stl/detail/store.hpp
              src/legate/experimental/stl/detail/launch_task.hpp
              src/legate/experimental/stl/detail/ranges.hpp
              src/legate/experimental/stl/detail/reduce.hpp
              src/legate/experimental/stl/detail/fill.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/experimental/stl/detail)

# Legate timing header
install(FILES src/legate/timing/timing.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/timing)

install(DIRECTORY ${LEGATE_DIR}/cmake/Modules/
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/legate" FILES_MATCHING
        PATTERN "*.cmake")

# MPI Wrapper
#
# We want to install the entire CMake project for the wrapper as-is into the share folder
# in the install path. That way -- once installed -- the user can build the wrapper
# themselves. This is also why we specifically avoided generating the install rules
install(FILES share/legate/mpi_wrapper/install.bash
              share/legate/mpi_wrapper/CMakeLists.txt
        DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/legate/mpi_wrapper")

install(FILES share/legate/mpi_wrapper/cmake/Config.cmake.in
        DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/legate/mpi_wrapper/cmake")

install(FILES share/legate/mpi_wrapper/src/legate_mpi_wrapper/mpi_wrapper.cc
              share/legate/mpi_wrapper/src/legate_mpi_wrapper/mpi_wrapper.h
              share/legate/mpi_wrapper/src/legate_mpi_wrapper/mpi_wrapper_types.h
        DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/legate/mpi_wrapper/src/legate_mpi_wrapper"
)

install(FILES share/legate/sanitizers/asan_default_options.txt
              share/legate/sanitizers/lsan_suppressions.txt
              share/legate/sanitizers/tsan_default_options.txt
              share/legate/sanitizers/tsan_suppressions.txt
              share/legate/sanitizers/ubsan_default_options.txt
        DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/legate/sanitizers")

include(cmake/Modules/utilities.cmake)

legate_install_from_tree(SRC_ROOT share COMMON_PATH legate/examples/binding/manual
                         DEST_ROOT "${CMAKE_INSTALL_DATAROOTDIR}"
                         FILES CMakeLists.txt hello_world.cc hello_world.py)

legate_install_from_tree(SRC_ROOT share
                         COMMON_PATH legate/examples/binding/cython
                         DEST_ROOT "${CMAKE_INSTALL_DATAROOTDIR}"
                         FILES hello_world.cc hello_world.h hello_world_cython.pyx
                               hello_world.py pyproject.toml CMakeLists.txt)

legate_install_from_tree(SRC_ROOT share
                         COMMON_PATH legate/examples/binding/pybind11
                         DEST_ROOT "${CMAKE_INSTALL_DATAROOTDIR}"
                         FILES hello_world.cc hello_world.py pyproject.toml
                               CMakeLists.txt)

include(${LEGATE_DIR}/cmake/Modules/debug_symbols.cmake)

legate_debug_syms(legate INSTALL_DIR ${lib_dir})

# ########################################################################################
# * install export -----------------------------------------------------------

set(doc_string
    [=[
Provide targets for Legate, the Foundation for All Legate Libraries.

Imported Targets:
  - legate::legate

]=])

file(READ ${LEGATE_DIR}/cmake/legate_helper_functions.cmake helper_functions)

# Normally this is done transparently (via the "code_string" below, embedded in the
# Findlegate.cmake) if the CMakeLists.txt calling this one finds the legate via a
# find_package() call. But if we are being built as a subdirectory, then we need to
# explicitly set(<the_variable> ... PARENT_SCOPE) in order for downstream to see it...
if(NOT PROJECT_IS_TOP_LEVEL)
  # These must match the decls below BEGIN MUST MATCH
  set(Legion_USE_CUDA "${Legion_USE_CUDA}" PARENT_SCOPE)
  set(Legion_USE_OpenMP "${Legion_USE_OpenMP}" PARENT_SCOPE)
  set(Legion_USE_Python "${Legion_USE_Python}" PARENT_SCOPE)
  set(Legion_CUDA_ARCH "${Legion_CUDA_ARCH}" PARENT_SCOPE)
  set(Legion_NETWORKS "${Legion_NETWORKS}" PARENT_SCOPE)
  set(Legion_BOUNDS_CHECKS "${Legion_BOUNDS_CHECKS}" PARENT_SCOPE)
  set(Legion_MAX_DIM "${Legion_MAX_DIM}" PARENT_SCOPE)
  set(Legion_MAX_FIELDS "${Legion_MAX_FIELDS}" PARENT_SCOPE)
  # END MUST MATCH
endif()

string(JOIN
       "\n"
       code_string
       [=[
if(NOT TARGET CCCL::Thrust)
  thrust_create_target(CCCL::Thrust FROM_OPTIONS)
endif()
]=]
       # These must match the decls above BEGIN MUST MATCH
       "set(Legion_USE_CUDA ${Legion_USE_CUDA})"
       "set(Legion_USE_OpenMP ${Legion_USE_OpenMP})"
       "set(Legion_USE_Python ${Legion_USE_Python})"
       "set(Legion_CUDA_ARCH ${Legion_CUDA_ARCH})"
       "set(Legion_NETWORKS ${Legion_NETWORKS})"
       "set(Legion_BOUNDS_CHECKS ${Legion_BOUNDS_CHECKS})"
       "set(Legion_MAX_DIM ${Legion_MAX_DIM})"
       "set(Legion_MAX_FIELDS ${Legion_MAX_FIELDS})"
       # END MUST MATCH
       [=[
if(Legion_NETWORKS)
  find_package(MPI REQUIRED COMPONENTS CXX)
endif()
]=]
       "${helper_functions}")

get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)

list(REMOVE_ITEM languages NONE)

message(STATUS "Enabled languages: ${languages}")

# FIXME(wonchanl): Passing LANGUAGES triggers a bug in rapids-cmake. Put it back once we
# bump the rapids-cmake version.
rapids_export(INSTALL legate
              EXPORT_SET legate-exports
              GLOBAL_TARGETS legate
              NAMESPACE legate::
              DOCUMENTATION doc_string
              FINAL_CODE_BLOCK code_string)

# build export targets
# FIXME(wonchanl): Passing LANGUAGES triggers a bug in rapids-cmake. Put it back once we
# bump the rapids-cmake version.
rapids_export(BUILD legate
              EXPORT_SET legate-exports
              GLOBAL_TARGETS legate
              NAMESPACE legate::
              DOCUMENTATION doc_string
              FINAL_CODE_BLOCK code_string)

# Symlink the module directory into the binary dir, so that the helper functions in
# legate-config.cmake can be used even if the project is not installed.
message(STATUS "Symlinking cmake module directory into ${CMAKE_CURRENT_BINARY_DIR}")
file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules
     ${CMAKE_CURRENT_BINARY_DIR}/Modules SYMBOLIC)

set(legate_ROOT ${CMAKE_CURRENT_BINARY_DIR})

list(APPEND legate_TIDY_SOURCES ${legate_SOURCES})

if(legate_BUILD_TESTS)
  include(CTest)

  add_subdirectory(tests/cpp)
endif()

if(legate_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(legate_BUILD_BENCHMARKS)
  add_subdirectory(${LEGATE_DIR}/benchmarks)
endif()

include(cmake/Modules/clang_tidy.cmake)

legate_add_tidy_target(SOURCES ${legate_TIDY_SOURCES})

# Legion sets this to "OFF" if not enabled, normalize it to an empty list instead
if(NOT Legion_NETWORKS)
  set(Legion_NETWORKS "")
endif()

add_custom_target(generate_install_info_py ALL
                  COMMAND ${CMAKE_COMMAND} -DLegion_NETWORKS="${Legion_NETWORKS}"
                          -DGASNet_CONDUIT="${GASNet_CONDUIT}"
                          -DLegion_USE_CUDA="${Legion_USE_CUDA}"
                          -DLegion_USE_OpenMP="${Legion_USE_OpenMP}"
                          -DLegion_MAX_DIM="${Legion_MAX_DIM}"
                          -DLegion_MAX_FIELDS="${Legion_MAX_FIELDS}"
                          -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
                          -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                          -DLEGATE_DIR="${LEGATE_DIR}" -DLEGATE_ARCH="${LEGATE_ARCH}"
                          -Dlegate_LIB_NAME="$<TARGET_FILE_PREFIX:legate::legate>$<TARGET_FILE_BASE_NAME:legate::legate>"
                          -Dlegate_FULL_LIB_NAME="$<TARGET_FILE_NAME:legate::legate>" -P
                          "${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate_install_info_py.cmake"
                  DEPENDS "${LEGATE_DIR}/legate/install_info.py.in"
                  BYPRODUCTS ${CMAKE_CURRENT_SOURCE_DIR}/legate/install_info.py
                  COMMENT "Generate install_info.py")

# touch these variables so they are not marked as "unused"
set(legate_maybe_ignored_variables_
    "${legate_CMAKE_PRESET_NAME};${CMAKE_BUILD_PARALLEL_LEVEL};")
if(NOT Legion_USE_CUDA)
  list(APPEND legate_maybe_ignored_variables_ "${legate_CUDA_FLAGS}")
  list(APPEND legate_maybe_ignored_variables_ "${CMAKE_CUDA_FLAGS_DEBUG}")
  list(APPEND legate_maybe_ignored_variables_ "${CMAKE_CUDA_FLAGS_RELEASE}")
endif()

include(cmake/Modules/uninstall.cmake)

legate_uninstall_target(TARGET uninstall)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
