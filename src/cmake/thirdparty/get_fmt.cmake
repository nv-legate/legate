#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

function(find_or_configure_fmt)
  list(APPEND CMAKE_MESSAGE_CONTEXT "fmt")

  include(${rapids-cmake-dir}/cpm/fmt.cmake)

  rapids_cpm_fmt(CPM_ARGS SYSTEM TRUE)
  cpm_export_variables(fmt)
endfunction()
