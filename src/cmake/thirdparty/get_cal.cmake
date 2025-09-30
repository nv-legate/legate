#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
