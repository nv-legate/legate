#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

function(find_or_configure_hdf5_vfd_gds)
  list(APPEND CMAKE_MESSAGE_CONTEXT "hdf5_vfd_gds")

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(hdf5_vfd_gds version git_url git_tag git_shallow
                             exclude_from_all)

  rapids_cpm_find(hdf5_vfd_gds "${version}"
                  CPM_ARGS
                  GIT_REPOSITORY "${git_url}"
                  GIT_SHALLOW "${git_shallow}" SYSTEM TRUE
                  GIT_TAG "${git_tag}"
                  EXCLUDE_FROM_ALL ${exclude_from_all}
                  OPTIONS "BUILD_TESTING OFF" "BUILD_EXAMPLES OFF"
                          "BUILD_DOCUMENTATION OFF")

  if(exclude_from_all)
    legate_install_dependencies(TARGETS hdf5_vfd_gds)
  endif()
  legate_export_variables(hdf5_vfd_gds)
endfunction()
