#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

# Use CPM to find or clone CCCL
function(find_or_configure_cccl)
  list(APPEND CMAKE_MESSAGE_CONTEXT "cccl")

  include(${rapids-cmake-dir}/cpm/cccl.cmake)

  rapids_cpm_cccl(BUILD_EXPORT_SET legate-exports INSTALL_EXPORT_SET legate-exports SYSTEM
                                                                                    TRUE)
  cpm_export_variables(CCCL)
endfunction()
