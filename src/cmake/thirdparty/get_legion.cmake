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

include_guard(GLOBAL)

function(legate_maybe_override_legion user_repository user_branch)
  # CPM_ARGS GIT_TAG and GIT_REPOSITORY don't do anything if you have already overridden
  # those options via a rapids_cpm_package_override() call. So we have to conditionally
  # override the defaults (by creating a temporary json file in build dir) only if the
  # user sets them.

  # See https://github.com/rapidsai/rapids-cmake/issues/575. Specifically, this function
  # is pretty much identical to
  # https://github.com/rapidsai/rapids-cmake/issues/575#issuecomment-2045374410.
  cmake_path(SET legion_overrides_json NORMALIZE
             "${LEGATE_CMAKE_DIR}/versions/legion_version.json")
  if(user_repository OR user_branch)
    # The user has set either one of these, time to create our cludge.
    file(READ "${legion_overrides_json}" default_legion_json)
    set(new_legion_json "${default_legion_json}")

    if(user_repository)
      string(JSON new_legion_json SET "${new_legion_json}" "packages" "Legion" "git_url"
             "\"${user_repository}\"")
    endif()

    if(user_branch)
      string(JSON new_legion_json SET "${new_legion_json}" "packages" "Legion" "git_tag"
             "\"${user_branch}\"")
    endif()

    string(JSON eq_json EQUAL "${default_legion_json}" "${new_legion_json}")
    if(NOT eq_json)
      cmake_path(SET legion_overrides_json NORMALIZE
                 "${CMAKE_CURRENT_BINARY_DIR}/legion_version.json")
      file(WRITE "${legion_overrides_json}" "${new_legion_json}")
    endif()
  endif()
  # https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_cpm_package_override/
  #
  # > Added in version v23.10.00: When the variable CPM_<package_name>_SOURCE exists, any
  # > override entries for package_name will be ignored.
  #
  # Work around this by unsetting this variable temporarily
  if(CPM_Legion_SOURCE)
    set(legion_src_cache_bkp "$CACHE{CPM_Legion_SOURCE}")
    unset(CPM_Legion_SOURCE)
    unset(CPM_Legion_SOURCE CACHE)
  endif()
  rapids_cpm_package_override("${legion_overrides_json}")
  if(legion_src_cache_bkp)
    # Don't need to reset the regular variable since we are in a function
    set(CPM_Legion_SOURCE "${legion_src_cache_bkp}" CACHE PATH "" FORCE)
  endif()
endfunction()

function(find_or_configure_legion_impl version git_repo git_branch shallow
         exclude_from_all)
  include("${LEGATE_CMAKE_DIR}/Modules/cpm_helpers.cmake")
  get_cpm_git_args(legion_cpm_git_args REPOSITORY "${git_repo}" BRANCH "${git_branch}"
                   SHALLOW "${shallow}")

  string(REGEX REPLACE "0([0-9]+)" "\\1" version "${version}")

  # cmake-lint: disable=W0106
  set_ifndef(Legion_PYTHON_EXTRA_INSTALL_ARGS
             "--root / --prefix \"\${CMAKE_INSTALL_PREFIX}\"")

  # Support comma and semicolon delimited lists
  string(REPLACE "," " " Legion_PYTHON_EXTRA_INSTALL_ARGS
                 "${Legion_PYTHON_EXTRA_INSTALL_ARGS}")
  string(REPLACE ";" " " Legion_PYTHON_EXTRA_INSTALL_ARGS
                 "${Legion_PYTHON_EXTRA_INSTALL_ARGS}")

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

  message(VERBOSE "legate: Legion version: ${version}")
  message(VERBOSE "legate: Legion git_repo: ${git_repo}")
  message(VERBOSE "legate: Legion git_branch: ${git_branch}")
  message(VERBOSE "legate: Legion exclude_from_all: ${exclude_from_all}")

  rapids_cpm_find(Legion ${version}
                  BUILD_EXPORT_SET legate-exports
                  INSTALL_EXPORT_SET legate-exports
                  GLOBAL_TARGETS Legion::Realm Legion::Regent Legion::Legion
                                 Legion::RealmRuntime Legion::LegionRuntime
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
                          "Legion_REDOP_HALF ON"
                          "Legion_REDOP_COMPLEX ON"
                          "Legion_UCX_DYNAMIC_LOAD ON"
                          # We never want local fields
                          "Legion_DEFAULT_LOCAL_FIELDS 0"
                          "Legion_HIJACK_CUDART OFF"
                          "CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON")

  set(Legion_VERSION "${version}" PARENT_SCOPE)
  set(Legion_GIT_REPO "${git_repo}" PARENT_SCOPE)
  set(Legion_GIT_BRANCH "${git_branch}" PARENT_SCOPE)
  set(Legion_USE_CUDA ${Legion_USE_CUDA} PARENT_SCOPE)
  set(Legion_USE_OpenMP ${Legion_USE_OpenMP} PARENT_SCOPE)
  set(Legion_USE_Python ${Legion_USE_Python} PARENT_SCOPE)
  set(Legion_CUDA_ARCH ${Legion_CUDA_ARCH} PARENT_SCOPE)
  set(Legion_BOUNDS_CHECKS ${Legion_BOUNDS_CHECKS} PARENT_SCOPE)
  set(Legion_NETWORKS ${Legion_NETWORKS} PARENT_SCOPE)
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

  legate_maybe_override_legion("${legate_LEGION_REPOSITORY}" "${legate_LEGION_BRANCH}")

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(Legion version git_repo git_branch shallow exclude_from_all)

  find_or_configure_legion_impl("${version}" "${git_repo}" "${git_branch}" "${shallow}"
                                "${exclude_from_all}")

  set(legion_vars
      Legion_VERSION Legion_GIT_REPO Legion_GIT_BRANCH Legion_USE_CUDA Legion_USE_OpenMP
      Legion_USE_Python Legion_CUDA_ARCH Legion_BOUNDS_CHECKS Legion_NETWORKS)
  foreach(var IN LISTS legion_vars)
    message(VERBOSE "${var}=${${var}}")
    set(${var} "${${var}}" PARENT_SCOPE)
  endforeach()
endfunction()
