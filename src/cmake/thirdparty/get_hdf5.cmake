#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

function(find_or_configure_hdf5)
  list(APPEND CMAKE_MESSAGE_CONTEXT "HDF5")

  rapids_find_package(HDF5 1.14.4 REQUIRED BUILD_EXPORT_SET legate-exports
                      INSTALL_EXPORT_SET legate-exports)
endfunction()
