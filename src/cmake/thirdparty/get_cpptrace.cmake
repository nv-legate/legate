#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

function(find_or_configure_cpptrace)
  list(APPEND CMAKE_MESSAGE_CONTEXT "cpptrace")

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(cpptrace version git_url git_tag git_shallow
                             exclude_from_all)

  set(cpptrace_options "CPPTRACE_BUILD_TESTING OFF" "CPPTRACE_BUILD_BENCHMARKING OFF")

  if(LEGATE_BUILD_PIP_WHEELS)
    # When building wheels we want to statically link cpptrace into legate, otherwise
    # auditwheel fails with "Cannot repair wheel, because required library
    # "libcpptrace.so.0" could not be located".
    list(APPEND cpptrace_options "BUILD_SHARED_LIBS OFF")
  endif()

  find_package(zstd)
  if(zstd_FOUND)
    # For whatever reason, cpptrace opts to download and install its own zstd instead of
    # using the system-provided one...
    list(APPEND cpptrace_options "CPPTRACE_USE_EXTERNAL_ZSTD ON")
  endif()

  find_package(libdwarf QUIET)
  if(libdwarf_FOUND)
    # ...same for libdwarf
    list(APPEND cpptrace_options "CPPTRACE_USE_EXTERNAL_LIBDWARF ON"
         "CPPTRACE_FIND_LIBDWARF_WITH_PKGCONFIG OFF")
  endif()

  if(APPLE)
    # libdwarf does not seem to be able to get line numbers or function names on macOS,
    # while addr2line does.
    list(APPEND cpptrace_options "CPPTRACE_GET_SYMBOLS_WITH_ADDR2LINE ON")
  endif()

  rapids_cpm_find(cpptrace "${version}"
                  GLOBAL_TARGETS cpptrace::cpptrace
                  CPM_ARGS
                  GIT_REPOSITORY "${git_url}"
                  GIT_SHALLOW "${git_shallow}" SYSTEM TRUE
                  GIT_TAG "${git_tag}"
                  EXCLUDE_FROM_ALL "${exclude_from_all}"
                  OPTIONS ${cpptrace_options})

  if(exclude_from_all)
    legate_install_dependencies(TARGETS cpptrace::cpptrace)
  endif()
  legate_export_variables(cpptrace)
endfunction()
