#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

function(find_or_configure_hdf5)
  list(APPEND CMAKE_MESSAGE_CONTEXT "HDF5")

  # Do not add HDF5 to the build or install export set. It is a private dependency, so
  # downstream users do not need to know about it (except for transitive link
  # dependencies).
  rapids_find_package(HDF5 1.14.4 REQUIRED)
endfunction()
