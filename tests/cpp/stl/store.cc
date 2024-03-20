/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "stl/stl.hpp"
#include "utilities/utilities.h"

#include <algorithm>
#include <gtest/gtest.h>

using STL = LegateSTLFixture;

namespace stl = legate::stl;

// NOLINTBEGIN(readability-magic-numbers)

TEST_F(STL, Test1DStore)
{
  auto store  = stl::create_store<std::int64_t>({4});
  auto pstore = stl::detail::get_logical_store(store).get_physical_store();
  auto read   = pstore.read_accessor<std::int64_t, 1>();
  EXPECT_EQ(store.dim(), 1);

  auto shape = store.extents();
  EXPECT_EQ(shape[0], 4);

  stl::fill(store, 0);

  EXPECT_EQ(4, std::distance(stl::elements_of(store).begin(), stl::elements_of(store).end()));
  EXPECT_EQ(4, std::count(stl::elements_of(store).begin(), stl::elements_of(store).end(), 0));

  auto store_span = stl::as_mdspan(store);
  store_span(0)   = 1;
  store_span(1)   = 2;
  store_span(2)   = 3;
  store_span(3)   = 4;

  EXPECT_EQ(1, read.read({0}));
  EXPECT_EQ(2, read.read({1}));
  EXPECT_EQ(3, read.read({2}));
  EXPECT_EQ(4, read.read({3}));
}

TEST_F(STL, Test2DStore)
{
  auto store  = stl::create_store<std::int64_t>({4, 5});
  auto pstore = stl::detail::get_logical_store(store).get_physical_store();
  auto read   = pstore.read_accessor<std::int64_t, 2>();
  EXPECT_EQ(store.dim(), 2);

  auto shape = store.extents();
  EXPECT_EQ(shape[0], 4);
  EXPECT_EQ(shape[1], 5);

  stl::fill(store, 0);

  EXPECT_EQ(20, std::distance(stl::elements_of(store).begin(), stl::elements_of(store).end()));
  EXPECT_EQ(20, std::count(stl::elements_of(store).begin(), stl::elements_of(store).end(), 0));

  auto store_span  = stl::as_mdspan(store);
  store_span(0, 0) = 1;
  store_span(0, 1) = 2;
  store_span(0, 2) = 3;
  store_span(0, 3) = 4;

  EXPECT_EQ(1, read.read({0, 0}));
  EXPECT_EQ(2, read.read({0, 1}));
  EXPECT_EQ(3, read.read({0, 2}));
  EXPECT_EQ(4, read.read({0, 3}));

  store_span(0, 1) = 11;
  store_span(1, 1) = 22;
  store_span(2, 1) = 33;
  store_span(3, 1) = 44;

  EXPECT_EQ(11, read.read({0, 1}));
  EXPECT_EQ(22, read.read({1, 1}));
  EXPECT_EQ(33, read.read({2, 1}));
  EXPECT_EQ(44, read.read({3, 1}));
}

TEST_F(STL, Test3DStore)
{
  auto store  = stl::create_store<std::int64_t>({4, 5, 6});
  auto pstore = stl::detail::get_logical_store(store).get_physical_store();
  auto read   = pstore.read_accessor<std::int64_t, 3>();
  EXPECT_EQ(store.dim(), 3);

  auto shape = store.extents();
  EXPECT_EQ(shape[0], 4);
  EXPECT_EQ(shape[1], 5);
  EXPECT_EQ(shape[2], 6);

  stl::fill(store, 0);

  EXPECT_EQ(120, std::distance(stl::elements_of(store).begin(), stl::elements_of(store).end()));
  EXPECT_EQ(120, std::count(stl::elements_of(store).begin(), stl::elements_of(store).end(), 0));

  auto store_span     = stl::as_mdspan(store);
  store_span(0, 0, 2) = 1;
  store_span(0, 1, 2) = 2;
  store_span(0, 2, 2) = 3;
  store_span(0, 3, 2) = 4;

  EXPECT_EQ(1, read.read({0, 0, 2}));
  EXPECT_EQ(2, read.read({0, 1, 2}));
  EXPECT_EQ(3, read.read({0, 2, 2}));
  EXPECT_EQ(4, read.read({0, 3, 2}));

  store_span(0, 1, 2) = 11;
  store_span(1, 1, 2) = 22;
  store_span(2, 1, 2) = 33;
  store_span(3, 1, 2) = 44;

  EXPECT_EQ(11, read.read({0, 1, 2}));
  EXPECT_EQ(22, read.read({1, 1, 2}));
  EXPECT_EQ(33, read.read({2, 1, 2}));
  EXPECT_EQ(44, read.read({3, 1, 2}));
}

TEST_F(STL, Constructors)
{
  static_assert(!std::is_default_constructible_v<stl::logical_store<int, 0>>);
  static_assert(!std::is_default_constructible_v<stl::logical_store<int, 1>>);
  static_assert(!std::is_default_constructible_v<stl::logical_store<int, 2>>);

  const stl::logical_store<int, 0> store0{{}};
  const stl::logical_store<int, 0> store1{{}, 42};
  const stl::logical_store store2{{}, 42};

  EXPECT_EQ(store2.dim(), 0);
  EXPECT_EQ(stl::as_mdspan(store2)(), 42);

  const stl::logical_store<int, 1> store3{{100}};
  const stl::logical_store<int, 1> store4{{100}, 42};
  const stl::logical_store store5{{100}, 42};

  EXPECT_EQ(store5.dim(), 1);
  EXPECT_EQ(store5.extents()[0], 100);
  EXPECT_EQ(stl::as_mdspan(store5)(0), 42);

  const stl::logical_store<int, 2> store6{{100, 200}};
  const stl::logical_store<int, 2> store7{{100, 200}, 42};
  const stl::logical_store store8{{100, 200}, 42};

  EXPECT_EQ(store8.dim(), 2);
  EXPECT_EQ(store8.extents()[0], 100);
  EXPECT_EQ(store8.extents()[1], 200);
  EXPECT_EQ(stl::as_mdspan(store8)(0, 0), 42);
}

// NOLINTEND(readability-magic-numbers)
