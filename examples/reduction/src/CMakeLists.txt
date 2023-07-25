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

legate_cpp_library_template(reduction TEMPLATE_SOURCES)

add_library(
  reduction
  bincount.cc
  categorize.cc
  histogram.cc
  matmul.cc
  mul.cc
  sum_over_axis.cc
  unique.cc
  ${TEMPLATE_SOURCES}
)

target_include_directories(reduction
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  INTERFACE
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(reduction PRIVATE legate::core)

