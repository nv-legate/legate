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

function(find_or_configure_google_benchmark)
  list(APPEND CMAKE_MESSAGE_CONTEXT "google_benchmark")

  if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    # Google benchmark considers any build that does not define NDEBUG to be a "debug"
    # build (and will emit warnings to that effect) even if CMAKE_BUILD_TYPE is "release".
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")
  endif()

  # Remove sanitizer flags from google benchmark, as these cause
  # ADDRESS_SANITIZER:DEADLYSIGNAL loops in CI.
  string(REGEX REPLACE [==[\-fsanitize=[^ ]+]==] "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

  include(${rapids-cmake-dir}/cpm/gbench.cmake)

  rapids_cpm_gbench(CPM_ARGS OPTIONS "BENCHMARK_ENABLE_TESTING OFF")
  cpm_export_variables(benchmark)
endfunction()
