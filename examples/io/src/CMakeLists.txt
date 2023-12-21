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

legate_cpp_library_template(legateio TEMPLATE_SOURCES)

add_library(
  legateio
  legateio.cc
  legateio.h
  read_file.cc
  read_even_tiles.cc
  read_uneven_tiles.cc
  util.cc
  write_file.cc
  write_uneven_tiles.cc
  write_even_tiles.cc
  ${TEMPLATE_SOURCES}
)

target_include_directories(legateio
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  INTERFACE
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(legateio PRIVATE legate::core)

