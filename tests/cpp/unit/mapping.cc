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

#include <gtest/gtest.h>

#include "core/mapping/detail/mapping.h"
#include "legate.h"

namespace unit {

TEST(Mapping, DimOrdering)
{
  // Default construct
  {
    auto order = legate::mapping::DimOrdering();
    EXPECT_EQ(order.kind(), legate::mapping::DimOrdering::Kind::C);
    EXPECT_EQ(order.dimensions(), std::vector<int32_t>());
  }

  // C ordering
  {
    auto c_order = legate::mapping::DimOrdering::c_order();
    EXPECT_EQ(c_order.kind(), legate::mapping::DimOrdering::Kind::C);
    EXPECT_EQ(c_order.dimensions(), std::vector<int32_t>());

    auto order = legate::mapping::DimOrdering();
    order.set_c_order();
    EXPECT_EQ(order.kind(), legate::mapping::DimOrdering::Kind::C);
  }

  // Fortran ordering
  {
    auto fortran_order = legate::mapping::DimOrdering::fortran_order();
    EXPECT_EQ(fortran_order.kind(), legate::mapping::DimOrdering::Kind::FORTRAN);
    EXPECT_EQ(fortran_order.dimensions(), std::vector<int32_t>());

    auto order = legate::mapping::DimOrdering();
    order.set_fortran_order();
    EXPECT_EQ(order.kind(), legate::mapping::DimOrdering::Kind::FORTRAN);
  }

  // custom ordering
  {
    std::vector<int32_t> dim = {0, 1, 2};
    auto custom_order        = legate::mapping::DimOrdering::custom_order(dim);
    EXPECT_EQ(custom_order.kind(), legate::mapping::DimOrdering::Kind::CUSTOM);
    EXPECT_EQ(custom_order.dimensions(), dim);

    auto order = legate::mapping::DimOrdering();
    order.set_custom_order(dim);
    EXPECT_EQ(order.kind(), legate::mapping::DimOrdering::Kind::CUSTOM);
    EXPECT_EQ(order.dimensions(), dim);
  }

  // custom ordering to c order
  {
    std::vector<int32_t> dim = {0, 1, 2};
    auto order               = legate::mapping::DimOrdering::custom_order(dim);
    order.set_c_order();
    EXPECT_EQ(order.kind(), legate::mapping::DimOrdering::Kind::C);
    EXPECT_EQ(order.dimensions(), std::vector<int32_t>());
  }
}

TEST(Mapping, InstanceMappingPolicy)
{
  // Empty InstanceMappingPolicy
  {
    auto policy = legate::mapping::InstanceMappingPolicy();
    EXPECT_EQ(policy.target, legate::mapping::StoreTarget::SYSMEM);
    EXPECT_EQ(policy.allocation, legate::mapping::AllocPolicy::MAY_ALLOC);
    EXPECT_EQ(policy.layout, legate::mapping::InstLayout::SOA);
    EXPECT_EQ(policy.ordering, legate::mapping::DimOrdering());
    EXPECT_EQ(policy.exact, false);
  }

  // Test set methods
  {
    auto target     = legate::mapping::StoreTarget::FBMEM;
    auto allocation = legate::mapping::AllocPolicy::MUST_ALLOC;
    auto layout     = legate::mapping::InstLayout::AOS;
    auto dim_order  = legate::mapping::DimOrdering::fortran_order();

    auto policy = legate::mapping::InstanceMappingPolicy();
    policy.set_target(target);
    EXPECT_EQ(policy, legate::mapping::InstanceMappingPolicy{}.with_target(target));

    policy.set_allocation_policy(allocation);
    EXPECT_EQ(policy,
              legate::mapping::InstanceMappingPolicy{}.with_target(target).with_allocation_policy(
                allocation));

    policy.set_instance_layout(layout);
    EXPECT_EQ(policy,
              legate::mapping::InstanceMappingPolicy{}
                .with_target(target)
                .with_allocation_policy(allocation)
                .with_instance_layout(layout));

    policy.set_ordering(dim_order);
    EXPECT_EQ(policy,
              legate::mapping::InstanceMappingPolicy{}
                .with_target(target)
                .with_allocation_policy(allocation)
                .with_instance_layout(layout)
                .with_ordering(dim_order));

    policy.set_exact(true);
    EXPECT_EQ(policy,
              legate::mapping::InstanceMappingPolicy{}
                .with_target(target)
                .with_allocation_policy(allocation)
                .with_instance_layout(layout)
                .with_ordering(dim_order)
                .with_exact(true));
  }

  // Test subsumes
  {
    auto judge_subsumes = [](auto policy_a, auto policy_b) {
      auto expect_result = true;
      if (!(policy_a.target == policy_b.target))
        expect_result = false;
      else if (!(policy_a.layout == policy_b.layout))
        expect_result = false;
      else if (!(policy_a.ordering == policy_b.ordering))
        expect_result = false;
      else if (!(policy_a.exact == policy_b.exact) && !policy_a.exact)
        expect_result = false;

      EXPECT_EQ(policy_a.subsumes(policy_b), expect_result);
    };

    auto target              = legate::mapping::StoreTarget::ZCMEM;
    auto allocation          = legate::mapping::AllocPolicy::MUST_ALLOC;
    auto layout              = legate::mapping::InstLayout::AOS;
    std::vector<int32_t> dim = {0, 1, 2};
    auto dim_order           = legate::mapping::DimOrdering::custom_order(dim);

    auto policy_a = legate::mapping::InstanceMappingPolicy{}
                      .with_target(target)
                      .with_allocation_policy(allocation)
                      .with_instance_layout(layout)
                      .with_ordering(dim_order)
                      .with_exact(true);
    auto policy_b = policy_a;
    judge_subsumes(policy_a, policy_b);

    for (int i = (int)legate::mapping::StoreTarget::SYSMEM;
         i <= (int)legate::mapping::StoreTarget::SOCKETMEM;
         ++i) {
      policy_b.set_target(static_cast<legate::mapping::StoreTarget>(i));
      judge_subsumes(policy_a, policy_b);
    }

    policy_b = policy_a;
    for (int i = (int)legate::mapping::AllocPolicy::MAY_ALLOC;
         i <= (int)legate::mapping::AllocPolicy::MUST_ALLOC;
         ++i) {
      policy_b.set_allocation_policy(static_cast<legate::mapping::AllocPolicy>(i));
      judge_subsumes(policy_a, policy_b);
    }

    policy_b = policy_a;
    for (int i = (int)legate::mapping::InstLayout::SOA; i <= (int)legate::mapping::InstLayout::AOS;
         ++i) {
      policy_b.set_instance_layout(static_cast<legate::mapping::InstLayout>(i));
      judge_subsumes(policy_a, policy_b);
    }

    policy_b = policy_a;
    policy_b.set_ordering(legate::mapping::DimOrdering::c_order());
    judge_subsumes(policy_a, policy_b);

    policy_b = policy_a;
    policy_b.set_exact(false);
    judge_subsumes(policy_a, policy_b);

    policy_a.set_exact(false);
    judge_subsumes(policy_a, policy_b);

    policy_b.set_exact(true);
    judge_subsumes(policy_a, policy_b);
  }
}

}  // namespace unit