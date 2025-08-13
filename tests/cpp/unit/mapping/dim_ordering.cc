/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace dim_ordering_unit {

namespace {

using DimOrderingTest = DefaultFixture;

void check_dim_ordering(const legate::mapping::DimOrdering& order,
                        legate::mapping::DimOrdering::Kind kind,
                        const std::vector<std::int32_t>& dim)
{
  ASSERT_EQ(order.kind(), kind);
  ASSERT_EQ(order.dimensions(), dim);
}

}  // namespace

TEST_F(DimOrderingTest, CreateDefault)
{
  const std::vector<std::int32_t> dim{};
  const legate::mapping::DimOrdering order{};
  check_dim_ordering(order, legate::mapping::DimOrdering::Kind::C, dim);
}

TEST_F(DimOrderingTest, CreateCOrder)
{
  const std::vector<std::int32_t> dim{};
  const auto c_order = legate::mapping::DimOrdering::c_order();
  check_dim_ordering(c_order, legate::mapping::DimOrdering::Kind::C, dim);
}

TEST_F(DimOrderingTest, CreateFortranOrder)
{
  const std::vector<std::int32_t> dim{};
  const auto fortran_order = legate::mapping::DimOrdering::fortran_order();
  check_dim_ordering(fortran_order, legate::mapping::DimOrdering::Kind::FORTRAN, dim);
}

TEST_F(DimOrderingTest, CreateCustomOrder)
{
  const std::vector<std::int32_t> dim_custom{0, 1, 2};
  const auto custom_order = legate::mapping::DimOrdering::custom_order(dim_custom);
  check_dim_ordering(custom_order, legate::mapping::DimOrdering::Kind::CUSTOM, dim_custom);
}

TEST_F(DimOrderingTest, SetCOrder)
{
  const std::vector<std::int32_t> dim{};
  legate::mapping::DimOrdering order{};

  order.set_c_order();
  check_dim_ordering(order, legate::mapping::DimOrdering::Kind::C, dim);
}

TEST_F(DimOrderingTest, SetFortranOrder)
{
  const std::vector<std::int32_t> dim{};
  legate::mapping::DimOrdering order{};

  order.set_fortran_order();
  check_dim_ordering(order, legate::mapping::DimOrdering::Kind::FORTRAN, dim);
}

TEST_F(DimOrderingTest, SetCustomOrder)
{
  const std::vector<std::int32_t> dim_custom{0, 1, 2};
  legate::mapping::DimOrdering order{};

  order.set_custom_order(dim_custom);
  check_dim_ordering(order, legate::mapping::DimOrdering::Kind::CUSTOM, dim_custom);
}

TEST_F(DimOrderingTest, SetOrderMultipleTimes)
{
  const std::vector<std::int32_t> dim{};
  const std::vector<std::int32_t> dim_custom{0, 1, 2};
  legate::mapping::DimOrdering order{};

  // Set to fortran, then custom, then back to c
  order.set_fortran_order();
  check_dim_ordering(order, legate::mapping::DimOrdering::Kind::FORTRAN, dim);

  order.set_custom_order(dim_custom);
  check_dim_ordering(order, legate::mapping::DimOrdering::Kind::CUSTOM, dim_custom);

  order.set_c_order();
  check_dim_ordering(order, legate::mapping::DimOrdering::Kind::C, dim);
}

TEST_F(DimOrderingTest, Equal)
{
  const legate::mapping::DimOrdering order1{};
  const legate::mapping::DimOrdering order2{};

  ASSERT_EQ(order1, order2);
}

TEST_F(DimOrderingTest, NotEqual)
{
  legate::mapping::DimOrdering order1{};
  legate::mapping::DimOrdering order2{};
  legate::mapping::DimOrdering order3{};

  order1.set_custom_order({});
  order2.set_fortran_order();
  order3.set_custom_order({0, 1});

  ASSERT_NE(order1, order2);
  ASSERT_NE(order1, order3);
  ASSERT_NE(order2, order3);
}

}  // namespace dim_ordering_unit
