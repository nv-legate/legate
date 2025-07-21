/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/mapping/detail/mapping.h>

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace unit {

namespace {

using DimOrderingTest           = DefaultFixture;
using InstanceMappingPolicyTest = DefaultFixture;

void check_dim_ordering(const legate::mapping::DimOrdering& order,
                        legate::mapping::DimOrdering::Kind kind,
                        const std::vector<std::int32_t>& dim)
{
  ASSERT_EQ(order.kind(), kind);
  ASSERT_EQ(order.dimensions(), dim);
}

}  // namespace

TEST_F(DimOrderingTest, Create)
{
  const std::vector<std::int32_t> dim{};

  // Create default
  const legate::mapping::DimOrdering order{};
  check_dim_ordering(order, legate::mapping::DimOrdering::Kind::C, dim);

  // Create c ordering
  const auto c_order = legate::mapping::DimOrdering::c_order();
  check_dim_ordering(c_order, legate::mapping::DimOrdering::Kind::C, dim);

  // Create fortran ordering
  const auto fortran_order = legate::mapping::DimOrdering::fortran_order();
  check_dim_ordering(fortran_order, legate::mapping::DimOrdering::Kind::FORTRAN, dim);

  // Create custom ordering
  const std::vector<std::int32_t> dim_custom{0, 1, 2};
  const auto custom_order = legate::mapping::DimOrdering::custom_order(dim_custom);
  check_dim_ordering(custom_order, legate::mapping::DimOrdering::Kind::CUSTOM, dim_custom);
}

TEST_F(DimOrderingTest, Set)
{
  const std::vector<std::int32_t> dim{};
  legate::mapping::DimOrdering order{};

  // Set c ordering
  order.set_c_order();
  check_dim_ordering(order, legate::mapping::DimOrdering::Kind::C, dim);

  // Set fortran ordering
  order.set_fortran_order();
  check_dim_ordering(order, legate::mapping::DimOrdering::Kind::FORTRAN, dim);

  // Set custom ordering
  const std::vector<std::int32_t> dim_custom{0, 1, 2};
  order.set_custom_order(dim_custom);
  check_dim_ordering(order, legate::mapping::DimOrdering::Kind::CUSTOM, dim_custom);

  // Set back to c ordering
  order.set_c_order();
  check_dim_ordering(order, legate::mapping::DimOrdering::Kind::C, dim);
}

TEST_F(InstanceMappingPolicyTest, Create)
{
  const legate::mapping::InstanceMappingPolicy policy{};

  ASSERT_EQ(policy.target, legate::mapping::StoreTarget::SYSMEM);
  ASSERT_EQ(policy.allocation, legate::mapping::AllocPolicy::MAY_ALLOC);
  ASSERT_EQ(policy.ordering, legate::mapping::DimOrdering{});
  ASSERT_EQ(policy.exact, false);
}

TEST_F(InstanceMappingPolicyTest, Set)
{
  constexpr auto target     = legate::mapping::StoreTarget::FBMEM;
  constexpr auto allocation = legate::mapping::AllocPolicy::MUST_ALLOC;
  const auto dim_order      = legate::mapping::DimOrdering::fortran_order();
  legate::mapping::InstanceMappingPolicy policy{};

  // Set target
  policy.set_target(target);
  ASSERT_EQ(policy, legate::mapping::InstanceMappingPolicy{}.with_target(target));

  // Set allocation policy
  policy.set_allocation_policy(allocation);
  ASSERT_EQ(policy,
            legate::mapping::InstanceMappingPolicy{}.with_target(target).with_allocation_policy(
              allocation));

  // Set ordering
  policy.set_ordering(dim_order);
  ASSERT_EQ(policy,
            legate::mapping::InstanceMappingPolicy{}
              .with_target(target)
              .with_allocation_policy(allocation)
              .with_ordering(dim_order));

  // Set exact
  policy.set_exact(true);
  ASSERT_EQ(policy,
            legate::mapping::InstanceMappingPolicy{}
              .with_target(target)
              .with_allocation_policy(allocation)
              .with_ordering(dim_order)
              .with_exact(true));
}

}  // namespace unit
