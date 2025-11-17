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
  # Clear the interface compile definitions as they incorrectly set
  # DNDEBUG,_FORTIFY_SOURCE=2. This should not be set publicly and causes issues in
  # consumers of the target.
  set_property(TARGET HDF5::HDF5 PROPERTY INTERFACE_COMPILE_DEFINITIONS "")
endfunction()
