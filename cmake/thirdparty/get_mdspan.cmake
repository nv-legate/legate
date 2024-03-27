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

function(find_or_configure_mdspan)
  if(CMAKE_CXX_STANDARD GREATER_EQUAL 23)
    include(CheckIncludeFileCXX)

    check_include_file_cxx("mdspan" have_std_mdspan)
    if(have_std_mdspan)
      return()
    endif()
  endif()

  rapids_cpm_find(mdspan 0.6
    BUILD_EXPORT_SET   legate-core-exports
    INSTALL_EXPORT_SET legate-core-exports
    CPM_ARGS
      GIT_REPOSITORY  https://github.com/kokkos/mdspan.git
      GIT_SHALLOW     TRUE
      SYSTEM          TRUE
      GIT_TAG         stable
      OPTIONS
        # Gotta set this, otherwise mdspan tries to guess a C++ standard
        "MDSPAN_CXX_STANDARD ${CMAKE_CXX_STANDARD}"
  )
endfunction()
