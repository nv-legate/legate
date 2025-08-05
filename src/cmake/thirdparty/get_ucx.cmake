#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

function(find_or_configure_ucx)
  list(APPEND CMAKE_MESSAGE_CONTEXT "ucx")

  # Try to find UCX using rapids_find_package.
  rapids_find_package(ucx REQUIRED BUILD_EXPORT_SET legate-exports
                      INSTALL_EXPORT_SET legate-exports)
endfunction()
