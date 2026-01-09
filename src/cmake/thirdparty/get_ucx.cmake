#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

function(find_or_configure_ucx)
  list(APPEND CMAKE_MESSAGE_CONTEXT "ucx")

  # Try to find UCX using rapids_find_package.
  #
  # Do not add UCX to the build or install export set. It is a private dependency, so
  # downstream users do not need to know about it (except for transitive link
  # dependencies).
  rapids_find_package(ucx REQUIRED)
endfunction()
