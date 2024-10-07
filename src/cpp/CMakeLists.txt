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

include("${LEGATE_CMAKE_DIR}/Modules/legate_options.cmake")

# ########################################################################################
# * Project definition -------------------------------------------------------

include(GNUInstallDirs)

# Write the version header
rapids_cmake_write_version_file(${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/version.h
                                PREFIX LEGATE)

include("${LEGATE_CMAKE_DIR}/Modules/compile_commands.cmake")

legate_export_compile_commands(DEST_DIR ${LEGATE_DIR})

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
set(legate_VERSIONS_JSON "${LEGATE_CMAKE_DIR}/versions/versions.json")
rapids_cpm_init(OVERRIDE ${legate_VERSIONS_JSON})

include("${LEGATE_CMAKE_DIR}/Modules/find_or_configure.cmake")

# ########################################################################################
# * Default Flags -------------------------------------------------------------

include("${LEGATE_CMAKE_DIR}/Modules/default_flags.cmake")

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

include("${LEGATE_CMAKE_DIR}/Modules/cuda_arch_helpers.cmake")

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

  add_subdirectory(${LEGATE_DIR}/share/legate/mpi_wrapper share/legate/mpi_wrapper)
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

include("${LEGATE_CMAKE_DIR}/Modules/generate_sanitizer_options.cmake")

legate_generate_sanitizer_options(
  SRC ${LEGATE_DIR}/share/legate/sanitizers/asan_default_options.txt
  DELIM [[:]]
  DEST "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/asan_default_options.h"
)

legate_generate_sanitizer_options(
  SRC ${LEGATE_DIR}/share/legate/sanitizers/lsan_suppressions.txt
  DELIM [[\n]]
  DEST "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/lsan_suppressions.h"
)

legate_generate_sanitizer_options(
  SRC ${LEGATE_DIR}/share/legate/sanitizers/ubsan_default_options.txt
  DELIM [[:]]
  DEST "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/ubsan_default_options.h"
)

legate_generate_sanitizer_options(
  SRC ${LEGATE_DIR}/share/legate/sanitizers/tsan_suppressions.txt
  DELIM [[\n]]
  DEST "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/tsan_suppressions.h"
)
legate_generate_sanitizer_options(
  SRC ${LEGATE_DIR}/share/legate/sanitizers/tsan_default_options.txt
  DELIM [[:]]
  DEST "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/tsan_default_options.h"
)

list(APPEND
     legate_SOURCES
     legate/comm/coll.cc
     legate/comm/detail/backend_network.cc
     legate/comm/detail/comm.cc
     legate/comm/detail/comm_cpu.cc
     legate/comm/detail/comm_local.cc
     legate/comm/detail/local_network.cc
     legate/comm/detail/logger.cc
     legate/comm/detail/thread_comm.cc
     legate/cuda/stream_pool.cc
     legate/cuda/detail/cuda_driver_api.cc
     legate/data/allocator.cc
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
     legate/operation/detail/req_analyzer.cc
     legate/operation/detail/scatter.cc
     legate/operation/detail/scatter_gather.cc
     legate/operation/detail/task.cc
     legate/operation/detail/task_launcher.cc
     legate/operation/detail/timing.cc
     legate/partitioning/constraint.cc
     legate/partitioning/detail/constraint.cc
     legate/partitioning/detail/constraint_solver.cc
     legate/partitioning/detail/partition.cc
     legate/partitioning/detail/partitioner.cc
     legate/partitioning/detail/partitioning_tasks.cc
     legate/partitioning/detail/restriction.cc
     legate/runtime/library.cc
     legate/runtime/runtime.cc
     legate/runtime/scope.cc
     legate/runtime/detail/communicator_manager.cc
     legate/runtime/detail/field_manager.cc
     legate/runtime/detail/library.cc
     legate/runtime/detail/partition_manager.cc
     legate/runtime/detail/projection.cc
     legate/runtime/detail/region_manager.cc
     legate/runtime/detail/runtime.cc
     legate/runtime/detail/shard.cc
     legate/runtime/detail/config.cc
     legate/runtime/detail/mapper_manager.cc
     legate/runtime/detail/argument_parsing.cc
     legate/task/registrar.cc
     legate/task/task.cc
     legate/task/task_context.cc
     legate/task/task_info.cc
     legate/task/variant_options.cc
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
     legate/type/type_info.cc
     legate/type/detail/type_info.cc
     legate/utilities/debug.cc
     legate/utilities/machine.cc
     legate/utilities/linearize.cc
     legate/utilities/internal_shared_ptr.cc
     legate/utilities/compiler.cc
     legate/utilities/abort.cc
     legate/utilities/detail/buffer_builder.cc
     legate/utilities/detail/env.cc
     legate/utilities/detail/tuple.cc
     legate/utilities/detail/deserializer.cc
     legate/utilities/detail/formatters.cc
     legate/timing/timing.cc
     # stl
     legate/experimental/stl/detail/clang_tidy_dummy.cpp)

if(legate_ENABLE_SANITIZERS)
  list(APPEND legate_SOURCES legate/utilities/detail/sanitizer_defaults.cc)
endif()

if(Legion_NETWORKS)
  list(APPEND legate_SOURCES legate/comm/detail/mpi_network.cc
       legate/comm/detail/mpi_interface.cc legate/comm/detail/comm_mpi.cc)
endif()

if(Legion_USE_OpenMP)
  list(APPEND legate_SOURCES legate/data/detail/array_tasks_omp.cc
       legate/partitioning/detail/partitioning_tasks_omp.cc)
endif()

if(Legion_USE_CUDA)
  list(APPEND legate_SOURCES legate/comm/detail/comm_nccl.cu
       legate/data/detail/array_tasks.cu legate/partitioning/detail/partitioning_tasks.cu)
  if(CAL_DIR)
    list(APPEND legate_SOURCES legate/comm/detail/comm_cal.cu)
  endif()
endif()

add_library(legate ${legate_SOURCES})
add_library(legate::legate ALIAS legate)

set(legate_CXX_PRIVATE_OPTIONS "")
set(legate_CUDA_PRIVATE_OPTIONS "")
set(legate_CXX_PUBLIC_OPTIONS "")
set(legate_CUDA_PUBLIC_OPTIONS "")
set(legate_LINKER_OPTIONS "")

include("${LEGATE_CMAKE_DIR}/Modules/set_cpu_arch_flags.cmake")

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

if(APPLE)
  # See https://github.com/NVIDIA/cccl/pull/2444
  target_compile_definitions(legate PUBLIC _LIBCUDACXX_STRING_H_HAS_CONST_OVERLOADS=1)
endif()

include("${LEGATE_CMAKE_DIR}/Modules/generate_legate_defines.cmake")

legate_generate_legate_defines()

set(legate_LOCAL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

target_include_directories(legate
                           PUBLIC $<BUILD_INTERFACE:${legate_LOCAL_INCLUDE_DIR}>
                                  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/legate>
                           INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/legate>
)

include("${LEGATE_CMAKE_DIR}/Modules/clang_tidy.cmake")

legate_add_tidy_diff_target()

foreach(src IN LISTS legate_SOURCES)
  legate_add_tidy_target(SOURCE "${src}")
endforeach()

if(legate_BUILD_DOCS)
  add_subdirectory(${LEGATE_DIR}/docs/legate docs/legate)
endif()

# ########################################################################################
# * install targets-----------------------------------------------------------

include(CPack)

rapids_cmake_install_lib_dir(lib_dir)

install(TARGETS legate DESTINATION ${lib_dir} EXPORT legate-exports)

install(FILES legate.h
              ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/legate/legate_defines.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/version.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate)

install(FILES legate/comm/coll_comm.h legate/comm/coll.h legate/comm/communicator.h
              legate/comm/communicator.inl
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/comm)

install(FILES legate/cuda/cuda.h legate/cuda/stream_pool.h legate/cuda/stream_pool.inl
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/cuda)

install(FILES legate/data/allocator.h
              legate/data/buffer.h
              legate/data/external_allocation.h
              legate/data/external_allocation.inl
              legate/data/inline_allocation.h
              legate/data/logical_array.h
              legate/data/logical_array.inl
              legate/data/logical_store.h
              legate/data/logical_store.inl
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
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/partitioning)

install(FILES legate/runtime/exception_mode.h legate/runtime/library.h
              legate/runtime/library.inl legate/runtime/resource.h
              legate/runtime/runtime.h legate/runtime/runtime.inl legate/runtime/scope.h
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
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/task)

install(FILES legate/type/type_info.h legate/type/type_info.inl legate/type/type_traits.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/type)

install(FILES legate/utilities/debug.h
              legate/utilities/debug.inl
              legate/utilities/dispatch.h
              legate/utilities/hash.h
              legate/utilities/machine.h
              legate/utilities/memory.h
              legate/utilities/memory.inl
              legate/utilities/nvtx_help.h
              legate/utilities/span.h
              legate/utilities/span.inl
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
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/legate/utilities/detail)

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

install(DIRECTORY ${LEGATE_CMAKE_DIR}/Modules/
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/legate" FILES_MATCHING
        PATTERN "*.cmake")

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

install(FILES ${LEGATE_DIR}/share/legate/mpi_wrapper/src/legate_mpi_wrapper/mpi_wrapper.cc
              ${LEGATE_DIR}/share/legate/mpi_wrapper/src/legate_mpi_wrapper/mpi_wrapper.h
              ${LEGATE_DIR}/share/legate/mpi_wrapper/src/legate_mpi_wrapper/mpi_wrapper_types.h
        DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/legate/mpi_wrapper/legate_mpi_wrapper")

install(FILES ${LEGATE_DIR}/share/legate/sanitizers/asan_default_options.txt
              ${LEGATE_DIR}/share/legate/sanitizers/lsan_suppressions.txt
              ${LEGATE_DIR}/share/legate/sanitizers/tsan_default_options.txt
              ${LEGATE_DIR}/share/legate/sanitizers/tsan_suppressions.txt
              ${LEGATE_DIR}/share/legate/sanitizers/ubsan_default_options.txt
        DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/legate/sanitizers")

# ########################################################################################
# * install Legate examples -----------------------------------------------------------

install(FILES ${LEGATE_DIR}/share/legate/examples/binding/manual/CMakeLists.txt
              ${LEGATE_DIR}/share/legate/examples/binding/manual/hello_world.cc
              ${LEGATE_DIR}/share/legate/examples/binding/manual/hello_world.py
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/legate/examples/binding/manual)

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

# ########################################################################################
# * install Legate debug symbols  --------------------------------------------------------

install(PROGRAMS ${LEGATE_DIR}/legate-bind.sh DESTINATION ${CMAKE_INSTALL_BINDIR})

# ########################################################################################
# * install Legate debug symbols ---------------------------------------------------------

include("${LEGATE_CMAKE_DIR}/Modules/debug_symbols.cmake")

legate_debug_syms(legate INSTALL_DIR ${lib_dir})

# ########################################################################################
# * install export -----------------------------------------------------------

set(doc_string
    [=[
Provide targets for Legate, the Foundation for All Legate Libraries.

Imported Targets:
  - legate::legate

]=])

file(READ "${LEGATE_CMAKE_DIR}/legate_helper_functions.cmake" helper_functions)

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
file(CREATE_LINK "${LEGATE_CMAKE_DIR}/Modules" ${CMAKE_CURRENT_BINARY_DIR}/Modules
     SYMBOLIC)

set(legate_ROOT ${CMAKE_CURRENT_BINARY_DIR})

if(legate_BUILD_TESTS)
  include(CTest)

  add_subdirectory(${LEGATE_DIR}/tests/cpp tests)
endif()

if(legate_BUILD_BENCHMARKS)
  add_subdirectory(${LEGATE_DIR}/benchmarks/cpp benchmarks)
endif()

# Legion sets this to "OFF" if not enabled, normalize it to an empty list instead
if(NOT Legion_NETWORKS)
  set(Legion_NETWORKS "")
endif()

add_custom_target(generate_install_info_py ALL
                  COMMAND ${CMAKE_COMMAND} -DLEGATE_CMAKE_DIR="${LEGATE_CMAKE_DIR}"
                          -DLEGATE_ARCH="${LEGATE_ARCH}"
                          -DLegion_NETWORKS="${Legion_NETWORKS}"
                          -DGASNet_CONDUIT="${GASNet_CONDUIT}"
                          -DLegion_USE_CUDA="${Legion_USE_CUDA}"
                          -DLegion_USE_OpenMP="${Legion_USE_OpenMP}"
                          -DLegion_MAX_DIM="${Legion_MAX_DIM}"
                          -DLegion_MAX_FIELDS="${Legion_MAX_FIELDS}"
                          -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
                          -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                          -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                          -DLEGATE_DIR="${LEGATE_DIR}" -DLEGATE_ARCH="${LEGATE_ARCH}"
                          -Dlegate_LIB_NAME="$<TARGET_FILE_PREFIX:legate::legate>$<TARGET_FILE_BASE_NAME:legate::legate>"
                          -Dlegate_FULL_LIB_NAME="$<TARGET_FILE_NAME:legate::legate>" -P
                          "${LEGATE_CMAKE_DIR}/generate_install_info_py.cmake"
                  DEPENDS "${LEGATE_CMAKE_DIR}/templates/install_info.py.in"
                  BYPRODUCTS "${LEGATE_DIR}/src/python/legate/install_info.py"
                  COMMENT "Generate install_info.py")

# touch these variables so they are not marked as "unused"
set(legate_maybe_ignored_variables_
    "${legate_CMAKE_PRESET_NAME};${CMAKE_BUILD_PARALLEL_LEVEL};")
if(NOT Legion_USE_CUDA)
  list(APPEND legate_maybe_ignored_variables_ "${legate_CUDA_FLAGS}")
  list(APPEND legate_maybe_ignored_variables_ "${CMAKE_CUDA_FLAGS_DEBUG}")
  list(APPEND legate_maybe_ignored_variables_ "${CMAKE_CUDA_FLAGS_RELEASE}")
endif()

include("${LEGATE_CMAKE_DIR}/Modules/uninstall.cmake")

legate_uninstall_target(TARGET uninstall)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
