/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace instance_mapping_policy_unit {

using InstanceMappingPolicyTest = DefaultFixture;

TEST_F(InstanceMappingPolicyTest, Create)
{
  const legate::mapping::InstanceMappingPolicy policy{};

  ASSERT_EQ(policy.target, legate::mapping::StoreTarget::SYSMEM);
  ASSERT_EQ(policy.allocation, legate::mapping::AllocPolicy::MAY_ALLOC);
  ASSERT_EQ(policy.ordering, legate::mapping::DimOrdering{});
  ASSERT_EQ(policy.exact, false);
}

TEST_F(InstanceMappingPolicyTest, SetTarget)
{
  constexpr auto target = legate::mapping::StoreTarget::FBMEM;
  legate::mapping::InstanceMappingPolicy policy{};

  policy.set_target(target);
  ASSERT_EQ(policy, legate::mapping::InstanceMappingPolicy{}.with_target(target));
}

TEST_F(InstanceMappingPolicyTest, SetAllocationPolicy)
{
  constexpr auto allocation = legate::mapping::AllocPolicy::MUST_ALLOC;
  legate::mapping::InstanceMappingPolicy policy{};

  policy.set_allocation_policy(allocation);
  ASSERT_EQ(policy, legate::mapping::InstanceMappingPolicy{}.with_allocation_policy(allocation));
}

TEST_F(InstanceMappingPolicyTest, SetOrdering)
{
  const auto dim_order = legate::mapping::DimOrdering::fortran_order();
  legate::mapping::InstanceMappingPolicy policy{};

  policy.set_ordering(dim_order);
  ASSERT_EQ(policy, legate::mapping::InstanceMappingPolicy{}.with_ordering(dim_order));
}

TEST_F(InstanceMappingPolicyTest, SetExact)
{
  legate::mapping::InstanceMappingPolicy policy{};

  policy.set_exact(true);
  ASSERT_EQ(policy, legate::mapping::InstanceMappingPolicy{}.with_exact(true));
}

TEST_F(InstanceMappingPolicyTest, SetMultipleProperties)
{
  constexpr auto target     = legate::mapping::StoreTarget::FBMEM;
  constexpr auto allocation = legate::mapping::AllocPolicy::MUST_ALLOC;
  const auto dim_order      = legate::mapping::DimOrdering::fortran_order();
  legate::mapping::InstanceMappingPolicy policy{};

  policy.set_target(target);
  policy.set_allocation_policy(allocation);
  policy.set_ordering(dim_order);
  policy.set_exact(true);
  ASSERT_EQ(policy,
            legate::mapping::InstanceMappingPolicy{}
              .with_target(target)
              .with_allocation_policy(allocation)
              .with_ordering(dim_order)
              .with_exact(true));
}

}  // namespace instance_mapping_policy_unit
