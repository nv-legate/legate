#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

function(find_or_configure_google_benchmark)
  list(APPEND CMAKE_MESSAGE_CONTEXT "google_benchmark")

  include(${rapids-cmake-dir}/cpm/gbench.cmake)

  rapids_cpm_gbench(CPM_ARGS "BENCHMARK_ENABLE_TESTING OFF")
endfunction()
