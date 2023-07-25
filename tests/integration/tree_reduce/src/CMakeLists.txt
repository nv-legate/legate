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

add_library(
  tree_reduce
  produce_normal.cc
  reduce_normal.cc
  produce_unbound.cc
  reduce_unbound.cc
  library.cc
)

target_include_directories(tree_reduce
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  INTERFACE
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(tree_reduce PRIVATE legate::core)

