#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

function(find_or_configure_kvikio)
  list(APPEND CMAKE_MESSAGE_CONTEXT "kvikio")

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(kvikio version git_url git_tag git_shallow exclude_from_all)

  # If the CUDA runtime library is static, we need to set the CUDA_STATIC_RUNTIME flag for
  # kvikio.
  if(CMAKE_CUDA_RUNTIME_LIBRARY STREQUAL "STATIC")
    set(CUDA_STATIC_RUNTIME TRUE)
  endif()
  rapids_cpm_find(kvikio "${version}"
                  CPM_ARGS
                  GIT_SHALLOW "${git_shallow}"
                  GIT_REPOSITORY "${git_url}" SYSTEM TRUE
                  GIT_TAG "${git_tag}" SOURCE_SUBDIR cpp
                  EXCLUDE_FROM_ALL ${exclude_from_all}
                  OPTIONS "KvikIO_BUILD_EXAMPLES OFF" "KvikIO_REMOTE_SUPPORT OFF")

  if(exclude_from_all)
    legate_install_dependencies(TARGETS kvikio::kvikio)
  endif()
  legate_export_variables(kvikio)
endfunction()
