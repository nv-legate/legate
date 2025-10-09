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

  # HACK: These should really be always-on (because cuPyNumeric uses half-precision), but
  # needed to work around CCCL compilation issues with
  # `cuda::std::atomic_ref<cuda::std::complex<__half>>` in legate-dataframe.
  #
  # We can remove this once https://github.com/nv-legate/legate.internal/pull/1418 lands.
  option(Legion_REDOP_HALF "Enable Half-precision reductions" ON)
  option(Legion_REDOP_COMPLEX "Enable complex reductions" ON)

  rapids_cpm_find(Legion "${version}"
                  BUILD_EXPORT_SET legate-exports
                  INSTALL_EXPORT_SET legate-exports
                  GLOBAL_TARGETS Legion::Regent Legion::Legion Legion::LegionRuntime
                  CPM_ARGS ${legion_cpm_git_args}
                           # HACK: Legion headers contain *many* warnings, but we would
                           # like to build with -Wall -Werror. But there is a work-around.
                           # Compilers treat system headers as special and do not emit any
                           # warnings about suspect code in them, so until legion cleans
                           # house, we mark their headers as "system" headers.
                           FIND_PACKAGE_ARGUMENTS
                           EXACT
                           SYSTEM
                           TRUE
                  EXCLUDE_FROM_ALL ${exclude_from_all}
                  OPTIONS "Legion_VERSION ${version}"
                          "Legion_REDOP_HALF ${Legion_REDOP_HALF}"
                          "Legion_REDOP_COMPLEX ${Legion_REDOP_COMPLEX}"
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
  set(Legion_VERSION "${version}" PARENT_SCOPE)
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

  # This was fixed in rapids-cmake 25.02
  if(CPM_Legion_SOURCE AND (rapids-cmake-version VERSION_LESS 25.02))
    # Need to make this warning nice and big because we already emit a bunch of warnings
    # like
    #
    # CMake Warning (dev) at /opt/homebrew/share/cmake/Modules/FetchContent.cmake:1953
    # (message):  Calling FetchContent_Populate(mdspan) is deprecated, call
    #
    # Because rapids-cmake hasn't bumped their CPM version yet. So we need this one to
    # stand out.`
    message(WARNING "===========================================================\n"
                    "                          WARNING\n"
                    "===========================================================\n"
                    "You have provided a source directory for Legion (${CPM_Legion_SOURCE}). "
                    "Legate requires that a series of patches are applied to Legion. This is "
                    "performed automatically by the build-system EXCEPT when a source directory "
                    "is provided. This is a limitation of the current build system and will be "
                    "fixed in a future release."
                    "\n"
                    "You must manually apply the patches under ${LEGATE_CMAKE_DIR}/patches/legion_*"
                    " before continuing."
                    "\n"
                    "===========================================================\n"
                    "                          WARNING\n"
                    "===========================================================")
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
