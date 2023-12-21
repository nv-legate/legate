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

legate_cpp_library_template(hello TEMPLATE_SOURCES)

add_library(
  hello
  hello_world.cc
  hello_world.h
  iota.cc
  sum.cc
  square.cc
  ${TEMPLATE_SOURCES}
)

target_include_directories(hello
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  INTERFACE
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(hello PRIVATE legate::core)

