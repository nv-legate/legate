#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

function(find_or_configure_argparse)
  list(APPEND CMAKE_MESSAGE_CONTEXT "argparse")

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(argparse version git_url git_tag git_shallow
                             exclude_from_all)

  # We don't add this package to BUILD_EXPORT_SET or INSTALL_EXPORT_SET because it is used
  # as a private, header-only dependency, and therefore no other package should need to
  # see any trace of it in cmake.
  rapids_cpm_find(argparse "${version}"
                  CPM_ARGS
                  GIT_REPOSITORY "${git_url}"
                  GIT_SHALLOW "${git_shallow}" SYSTEM TRUE
                  GIT_TAG "${git_tag}"
                  EXCLUDE_FROM_ALL ${exclude_from_all})

  if(exclude_from_all)
    legate_install_dependencies(TARGETS argparse::argparse)
  endif()
  legate_export_variables(argparse)
endfunction()
