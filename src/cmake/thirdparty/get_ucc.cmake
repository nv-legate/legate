#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

function(find_or_configure_ucc)
  list(APPEND CMAKE_MESSAGE_CONTEXT "ucc")

  # Try to find UCC using rapids_find_package. ucc find package has a bug that causes it
  # to fail if called twice. We are checking to avoid calling it twice.
  #
  # Do not add UCC/UCX to the build or install export set. It is a private dependency, so
  # downstream users do not need to know about it (except for transitive link
  # dependencies).
  if(NOT TARGET ucc::ucc)
    rapids_find_package(ucc REQUIRED)
  endif()
endfunction()
