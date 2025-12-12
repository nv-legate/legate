/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/experimental/stl.hpp>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

using STL = DefaultFixture;

namespace stl = legate::experimental::stl;

namespace {

// NOLINTBEGIN(readability-magic-numbers)

// clang-tidy complains that "2D" is not lower case
// NOLINTNEXTLINE(readability-identifier-naming)
void test_rows_of_2D_store()
{
  auto store      = stl::create_store<std::int64_t>({4, 5}, /*value=*/0);
  auto store_span = stl::as_mdspan(store);

  auto rows = stl::rows_of(store);
  EXPECT_EQ(rows.size(), 4);

  auto row = *rows.begin();
  static_assert(std::is_same_v<decltype(row), stl::logical_store<std::int64_t, 1>>);
  EXPECT_EQ(row.extents()[0], 5);

  // write through the row
  auto row_span = stl::as_mdspan(row);
  row_span[0]   = 1;
  row_span[1]   = 2;
  row_span[2]   = 3;
  row_span[3]   = 4;
  row_span[4]   = 5;

  EXPECT_EQ(store_span(0, 0), 1);
  EXPECT_EQ(store_span(0, 1), 2);
  EXPECT_EQ(store_span(0, 2), 3);
  EXPECT_EQ(store_span(0, 3), 4);
  EXPECT_EQ(store_span(0, 4), 5);

  row = *std::next(rows.begin());
  EXPECT_EQ(row.extents()[0], 5);

  // write through the row
  row_span    = stl::as_mdspan(row);
  row_span[0] = 11;
  row_span[1] = 22;
  row_span[2] = 33;
  row_span[3] = 44;
  row_span[4] = 55;

  EXPECT_EQ(store_span(1, 0), 11);
  EXPECT_EQ(store_span(1, 1), 22);
  EXPECT_EQ(store_span(1, 2), 33);
  EXPECT_EQ(store_span(1, 3), 44);
  EXPECT_EQ(store_span(1, 4), 55);
}

// clang-tidy complains that "2D" is not lower case
// NOLINTNEXTLINE(readability-identifier-naming)
void test_columns_of_2D_store()
{
  auto store                         = stl::create_store<std::int64_t>({4, 5}, /*value=*/0);
  auto store_span                    = stl::as_mdspan(store);
  const legate::PhysicalStore pstore = stl::detail::get_logical_store(store).get_physical_store();
  auto acc                           = pstore.read_accessor<std::int64_t, 2>();

  auto cols = stl::columns_of(store);
  EXPECT_EQ(cols.size(), 5);

  auto col = *cols.begin();
  static_assert(std::is_same_v<decltype(col), stl::logical_store<std::int64_t, 1>>);
  EXPECT_EQ(col.extents()[0], 4);

  const legate::PhysicalStore col_pstore = stl::detail::get_logical_store(col).get_physical_store();

  EXPECT_EQ(col_pstore.dim(), 1);
  EXPECT_EQ(col_pstore.shape<1>(), (legate::Rect<1>{Realm::make_point(0), Realm::make_point(3)}));
  auto col_acc = col_pstore.read_accessor<std::int64_t, 1>();

  // write through the col
  auto col_span = stl::as_mdspan(col);
  col_span[0]   = 1;
  col_span[1]   = 2;
  col_span[2]   = 3;
  col_span[3]   = 4;

  EXPECT_EQ(col_acc.read(Realm::make_point(0)), 1);
  EXPECT_EQ(col_acc.read(Realm::make_point(1)), 2);
  EXPECT_EQ(col_acc.read(Realm::make_point(2)), 3);
  EXPECT_EQ(col_acc.read(Realm::make_point(3)), 4);

  EXPECT_EQ(acc.read({0, 0}), 1);
  EXPECT_EQ(acc.read({1, 0}), 2);
  EXPECT_EQ(acc.read({2, 0}), 3);
  EXPECT_EQ(acc.read({3, 0}), 4);

  EXPECT_EQ(store_span(0, 0), 1);
  EXPECT_EQ(store_span(1, 0), 2);
  EXPECT_EQ(store_span(2, 0), 3);
  EXPECT_EQ(store_span(3, 0), 4);

  col = *std::next(cols.begin());
  EXPECT_EQ(col.extents()[0], 4);

  // write through the col
  col_span    = stl::as_mdspan(col);
  col_span[0] = 11;
  col_span[1] = 22;
  col_span[2] = 33;
  col_span[3] = 44;

  EXPECT_EQ(store_span(0, 1), 11);
  EXPECT_EQ(store_span(1, 1), 22);
  EXPECT_EQ(store_span(2, 1), 33);
  EXPECT_EQ(store_span(3, 1), 44);
}

// NOLINTEND(readability-magic-numbers)

}  // namespace

TEST_F(STL, TestRowsOf2DStore) { test_rows_of_2D_store(); }

TEST_F(STL, TestColumnsOf2DStore) { test_columns_of_2D_store(); }
