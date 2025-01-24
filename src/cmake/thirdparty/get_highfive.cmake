#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  cpm_export_variables(HighFive)
endfunction()
