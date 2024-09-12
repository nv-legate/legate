#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

macro(legate_ensure_legate)
  if(legate_CMAKE_PRESET_NAME AND NOT legate_ROOT)
    # If we are using a preset (and the user is not overriding the path anyways), then we
    # know exactly where the root is
    cmake_path(SET legate_ROOT NORMALIZE
               "${LEGATE_DIR}/build/${legate_CMAKE_PRESET_NAME}")
  endif()

  if(NOT (CMAKE_PROJECT_NAME STREQUAL "legate"))
    # If CMAKE_PROJECT_NAME is not legate, then we are not configuring from top-level.
    find_package(legate REQUIRED)
  endif()
endmacro()
