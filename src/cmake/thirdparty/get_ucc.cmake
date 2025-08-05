#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

function(find_or_configure_ucc)
  list(APPEND CMAKE_MESSAGE_CONTEXT "ucc")

  # Try to find UCC using rapids_find_package. ucc find package has a bug that causes it
  # to fail if called twice. We are checking to avoid calling it twice.
  if(NOT TARGET ucc::ucc)
    rapids_find_package(ucc REQUIRED BUILD_EXPORT_SET legate-exports
                        INSTALL_EXPORT_SET legate-exports)
  endif()
endfunction()
