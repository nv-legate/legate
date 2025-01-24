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

function(find_or_configure_cal)
  list(APPEND CMAKE_MESSAGE_CONTEXT "cal")

  if(NOT TARGET CAL::CAL)
    rapids_find_generate_module(CAL HEADER_NAMES cal.h LIBRARY_NAMES cal)
    # Currently CAL has no CMake build-system so we require it built and installed on the
    # machine already
    rapids_find_package(CAL REQUIRED)
  endif()
endfunction()
