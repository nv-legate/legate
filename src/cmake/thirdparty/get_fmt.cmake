#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

function(find_or_configure_fmt)
  list(APPEND CMAKE_MESSAGE_CONTEXT "fmt")

  include(${rapids-cmake-dir}/cpm/fmt.cmake)

  rapids_cpm_fmt(CPM_ARGS SYSTEM TRUE)
  cpm_export_variables(fmt)
endfunction()
