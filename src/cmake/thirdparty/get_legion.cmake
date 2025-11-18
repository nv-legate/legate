#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

function(find_or_configure_legion_impl version git_repo git_branch shallow
         exclude_from_all)
  include("${LEGATE_CMAKE_DIR}/Modules/cpm_helpers.cmake")
  get_cpm_git_args(legion_cpm_git_args REPOSITORY "${git_repo}" BRANCH "${git_branch}"
                   SHALLOW "${shallow}")

  string(REGEX REPLACE "0([0-9]+)" "\\1" version "${version}")

  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(Legion_BACKTRACE_USE_LIBDW ON)
  else()
    set(Legion_BACKTRACE_USE_LIBDW OFF)
  endif()

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Legion_CXX_FLAGS}")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${Legion_CUDA_FLAGS}")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${Legion_LINKER_FLAGS}")
  set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} ${Legion_LINKER_FLAGS}")

  if(BUILD_SHARED_LIBS)
    set(Legion_CUDA_DYNAMIC_LOAD ON)
  else()
    set_ifndef(Legion_CUDA_DYNAMIC_LOAD OFF)
  endif()

  # Maintain the default behavior of using the MPI bootstrap plugin for UCX.
  set(Legion_UCX_MPI_BOOTSTRAP ON)
  # If we are building wheels, we don't want to dynamically link against the CUDA runtime.
  if(LEGATE_BUILD_PIP_WHEELS)
    set(Legion_CUDA_DYNAMIC_LOAD OFF)
    set(Legion_UCX_MPI_BOOTSTRAP OFF)
  endif()

  set_ifndef(Legion_EMBED_GASNet_CONFIGURE_ARGS "--with-ibv-max-hcas=8")

  message(VERBOSE "legate: Legion version: ${version}")
  message(VERBOSE "legate: Legion git_repo: ${git_repo}")
  message(VERBOSE "legate: Legion git_branch: ${git_branch}")
  message(VERBOSE "legate: Legion exclude_from_all: ${exclude_from_all}")

  rapids_cpm_find(Legion "${version}"
                  BUILD_EXPORT_SET legate-exports
                  INSTALL_EXPORT_SET legate-exports
                  GLOBAL_TARGETS Legion::Regent Legion::Legion Legion::LegionRuntime
                  CPM_ARGS ${legion_cpm_git_args} SYSTEM TRUE
                  EXCLUDE_FROM_ALL ${exclude_from_all}
                  OPTIONS "Legion_REDOP_HALF OFF"
                          "Legion_REDOP_COMPLEX OFF"
                          "Legion_UCX_DYNAMIC_LOAD ON"
                          # We never want local fields
                          "Legion_DEFAULT_LOCAL_FIELDS 0"
                          "Legion_HIJACK_CUDART OFF"
                          "Legion_BUILD_BINDINGS OFF"
                          "Legion_EMBED_GASNet_CONFIGURE_ARGS ${Legion_EMBED_GASNet_CONFIGURE_ARGS}"
                          "Legion_UCX_MPI_BOOTSTRAP ${Legion_UCX_MPI_BOOTSTRAP}"
                          "Legion_USE_ZLIB OFF"
                          "Legion_CUDA_DYNAMIC_LOAD ${Legion_CUDA_DYNAMIC_LOAD}"
                          "CMAKE_INSTALL_BINDIR ${legate_DEP_INSTALL_BINDIR}"
                          "CMAKE_INSTALL_INCLUDEDIR ${legate_DEP_INSTALL_INCLUDEDIR}"
                          "INSTALL_SUFFIX -legate"
                          "CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON")

  legate_export_variables(Legion)
  set_parent_scope(Legion_VERSION)
  set(Legion_GIT_REPO "${git_repo}" PARENT_SCOPE)
  set(Legion_GIT_BRANCH "${git_branch}" PARENT_SCOPE)
  set_parent_scope(Legion_USE_CUDA)
  set_parent_scope(Legion_USE_OpenMP)
  set_parent_scope(Legion_USE_Python)
  set_parent_scope(Legion_CUDA_ARCH)
  set_parent_scope(Legion_BOUNDS_CHECKS)
  set_parent_scope(Legion_NETWORKS)
endfunction()

function(find_or_configure_legion)
  list(APPEND CMAKE_MESSAGE_CONTEXT "legion")

  if(Legion_HIJACK_CUDART)
    message(FATAL_ERROR [=[
#####################################################################
Error: Realm's CUDA runtime hijack is incompatible with NCCL.
Please note that your code will crash catastrophically as soon as it
calls into NCCL either directly or through some other Legate library.
#####################################################################
]=])
  endif()

  legate_maybe_override_package_info(Legion "${legate_LEGION_BRANCH}")
  legate_load_overrideable_package_info(Legion version git_repo git_branch shallow
                                        exclude_from_all)
  if(CPM_Legion_SOURCE)
    # The user is supplying a source directory, relax version requirement.
    message(STATUS "User supplied Legion source directory")
    set(version "0.0.0")
  endif()
  find_or_configure_legion_impl("${version}" "${git_repo}" "${git_branch}" "${shallow}"
                                "${exclude_from_all}")

  set(legion_vars
      Legion_VERSION Legion_GIT_REPO Legion_GIT_BRANCH Legion_USE_CUDA Legion_USE_OpenMP
      Legion_USE_Python Legion_CUDA_ARCH Legion_BOUNDS_CHECKS Legion_NETWORKS)
  foreach(var IN LISTS legion_vars)
    message(VERBOSE "${var}=${${var}}")
    set_parent_scope(${var})
  endforeach()
  legate_export_variables(Legion)
endfunction()
