/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace test_child_store {

using ChildStore = DefaultFixture;

namespace {

constexpr std::size_t EXTENT    = 7;
constexpr std::size_t TILE_SIZE = 4;
constexpr std::int64_t FACTOR   = 5;

}  // namespace

TEST_F(ChildStore, Simple)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Shape{EXTENT, EXTENT}, legate::int64());
  auto part    = store.partition_by_tiling({TILE_SIZE, TILE_SIZE});

  runtime->issue_fill(store, legate::Scalar{std::int64_t{-1}});
  for (std::uint64_t dim0 = 0; dim0 < 2; ++dim0) {
    for (std::uint64_t dim1 = 0; dim1 < 2; ++dim1) {
      auto child   = part.get_child_store(legate::tuple<std::uint64_t>{dim0, dim1});
      auto extents = child.extents();
      std::cout << "Child store (" << dim0 << "," << dim1 << ") " << child.to_string() << '\n';
      EXPECT_EQ(extents[0], dim0 == 0 ? TILE_SIZE : EXTENT - TILE_SIZE);
      EXPECT_EQ(extents[1], dim1 == 0 ? TILE_SIZE : EXTENT - TILE_SIZE);

      auto p_child = child.get_physical_store();
      auto acc     = p_child.write_accessor<std::int64_t, 2>();
      for (std::int64_t i = 0; i < static_cast<std::int64_t>(extents[0]); ++i) {
        for (std::int64_t j = 0; j < static_cast<std::int64_t>(extents[1]); ++j) {
          acc[{i, j}] = static_cast<legate::coord_t>((dim0 * FACTOR) + dim1);
        }
      }
    }
  }

  auto p_store = store.get_physical_store();
  auto acc     = p_store.read_accessor<std::int64_t, 2>();

  for (std::int64_t i = 0; i < static_cast<std::int64_t>(EXTENT); ++i) {
    for (std::int64_t j = 0; j < static_cast<std::int64_t>(EXTENT); ++j) {
      auto expected_value = ((i / TILE_SIZE) * FACTOR) + (j / TILE_SIZE);
      EXPECT_EQ((acc[{i, j}]), expected_value);
    }
  }
}

}  // namespace test_child_store
