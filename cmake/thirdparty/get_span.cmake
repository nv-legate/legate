#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

function(find_or_configure_span)
  list(APPEND CMAKE_MESSAGE_CONTEXT "span")

  if(CMAKE_CXX_STANDARD GREATER_EQUAL 20)
    include(CheckIncludeFileCXX)

    check_include_file_cxx("span" have_std_span)
    if(have_std_span)
      return()
    endif()
  endif()

  legate_core_parse_versions_json(
    PACKAGE     span
    VERSION     version
    GIT_URL     git_url
    GIT_SHALLOW git_shallow
    GIT_TAG     git_tag
  )

  rapids_cpm_find(span "${version}"
    BUILD_EXPORT_SET   legate-core-exports
    INSTALL_EXPORT_SET legate-core-exports
    CPM_ARGS
      GIT_REPOSITORY  "${git_url}"
      GIT_SHALLOW     "${git_shallow}"
      SYSTEM          TRUE
      GIT_TAG         "${git_tag}"
   )
endfunction()
