#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

function(find_or_configure_highfive)
  list(APPEND CMAKE_MESSAGE_CONTEXT "highfive")

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(HighFive version git_url git_tag git_shallow
                             exclude_from_all)

  rapids_cpm_find(HighFive "${version}"
                  CPM_ARGS
                  GIT_REPOSITORY "${git_url}"
                  GIT_SHALLOW "${git_shallow}" SYSTEM TRUE
                  GIT_TAG "${git_tag}"
                  EXCLUDE_FROM_ALL ${exclude_from_all}
                  OPTIONS "USE_BOOST OFF" "HIGHFIVE_EXAMPLES OFF"
                          "HIGHFIVE_BUILD_DOCS OFF" "HIGHFIVE_HAS_CONCEPTS OFF")

  if(exclude_from_all)
    legate_install_dependencies(TARGETS HighFive::HighFive)
  endif()
  legate_export_variables(HighFive)
endfunction()
