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

function(find_or_configure_kvikio)
  list(APPEND CMAKE_MESSAGE_CONTEXT "kvikio")

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(kvikio version git_url git_tag git_shallow exclude_from_all)

  rapids_cpm_find(kvikio "${version}"
                  CPM_ARGS
                  GIT_SHALLOW "${git_shallow}"
                  GIT_REPOSITORY "${git_url}" SYSTEM TRUE
                  GIT_TAG "${git_tag}" SOURCE_SUBDIR cpp
                  EXCLUDE_FROM_ALL ${exclude_from_all}
                  OPTIONS "KvikIO_BUILD_EXAMPLES OFF")

  if(exclude_from_all)
    legate_install_dependencies(TARGETS kvikio::kvikio)
  endif()
  cpm_export_variables(kvikio)
endfunction()
