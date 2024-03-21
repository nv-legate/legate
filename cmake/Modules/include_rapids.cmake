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

macro(legate_include_rapids)
  if(NOT rapids-cmake-version)
    # default
    set(rapids-cmake-version 24.02)
    set(rapids-cmake-sha "6d9ab304e8d561025de7c17a1b081440a8d9f8fc")
  endif()
  if (NOT _LEGATE_HAS_RAPIDS)
    if(NOT EXISTS ${CMAKE_BINARY_DIR}/LEGATE_RAPIDS.cmake)
      file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-${rapids-cmake-version}/RAPIDS.cmake
           ${CMAKE_BINARY_DIR}/LEGATE_RAPIDS.cmake)
    endif()
    include(${CMAKE_BINARY_DIR}/LEGATE_RAPIDS.cmake)
    include(rapids-cmake)
    include(rapids-cpm)
    include(rapids-cuda)
    include(rapids-export)
    include(rapids-find)
    set(_LEGATE_HAS_RAPIDS ON)
  endif()
endmacro()
