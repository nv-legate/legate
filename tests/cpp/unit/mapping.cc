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

using DimOrderingTest              = DefaultFixture;
using InstanceMappingPolicyTest    = DefaultFixture;
using InstanceMappingPolicySubsume = DefaultFixture;

class StoreTargetInput : public DefaultFixture,
                         public ::testing::WithParamInterface<legate::mapping::StoreTarget> {};

class AllocationInput : public DefaultFixture,
                        public ::testing::WithParamInterface<legate::mapping::AllocPolicy> {};

class InstanceInput : public DefaultFixture,
                      public ::testing::WithParamInterface<legate::mapping::InstLayout> {};

class DimOrderInput : public DefaultFixture,
                      public ::testing::WithParamInterface<legate::mapping::DimOrdering> {};

class ExtractInput : public DefaultFixture,
                     public ::testing::WithParamInterface<std::tuple<bool, bool>> {};

INSTANTIATE_TEST_SUITE_P(InstanceMappingPolicySubsume,
                         StoreTargetInput,
                         ::testing::Values(legate::mapping::StoreTarget::SYSMEM,
                                           legate::mapping::StoreTarget::FBMEM,
                                           legate::mapping::StoreTarget::ZCMEM,
                                           legate::mapping::StoreTarget::SOCKETMEM));

INSTANTIATE_TEST_SUITE_P(InstanceMappingPolicySubsume,
                         AllocationInput,
                         ::testing::Values(legate::mapping::AllocPolicy::MAY_ALLOC,
                                           legate::mapping::AllocPolicy::MUST_ALLOC));

INSTANTIATE_TEST_SUITE_P(InstanceMappingPolicySubsume,
                         InstanceInput,
                         ::testing::Values(legate::mapping::InstLayout::SOA,
                                           legate::mapping::InstLayout::AOS));

INSTANTIATE_TEST_SUITE_P(InstanceMappingPolicySubsume,
                         DimOrderInput,
                         ::testing::Values(legate::mapping::DimOrdering::c_order(),
                                           legate::mapping::DimOrdering::fortran_order(),
                                           legate::mapping::DimOrdering::custom_order(
                                             std::vector<std::int32_t>{1, 2, 3})));

INSTANTIATE_TEST_SUITE_P(InstanceMappingPolicySubsume,
                         ExtractInput,
                         ::testing::Combine(::testing::Bool(), ::testing::Bool()));

void check_dim_ordering(const legate::mapping::DimOrdering& order,
                        legate::mapping::DimOrdering::Kind kind,
                        const std::vector<std::int32_t>& dim)
{
  ASSERT_EQ(order.kind(), kind);
  ASSERT_EQ(order.dimensions(), dim);
}

void check_subsume(const legate::mapping::InstanceMappingPolicy& policy_a,
                   const legate::mapping::InstanceMappingPolicy& policy_b)
{
  // We expect policy_a subsumes policy_b if any of below condition is met:
  // 1) their target, layout, ordering, exact are same;
  // 2) their target, layout, ordering are same and policy_a's exact is true.
  auto expect_result = (policy_a.target == policy_b.target && policy_a.layout == policy_b.layout &&
                        policy_a.ordering == policy_b.ordering &&
                        ((policy_a.exact == policy_b.exact) || policy_a.exact));

  ASSERT_EQ(policy_a.subsumes(policy_b), expect_result);
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
  ASSERT_EQ(policy.layout, legate::mapping::InstLayout::SOA);
  ASSERT_EQ(policy.ordering, legate::mapping::DimOrdering{});
  ASSERT_EQ(policy.exact, false);
}

TEST_F(InstanceMappingPolicyTest, Set)
{
  constexpr auto target     = legate::mapping::StoreTarget::FBMEM;
  constexpr auto allocation = legate::mapping::AllocPolicy::MUST_ALLOC;
  constexpr auto layout     = legate::mapping::InstLayout::AOS;
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

  // Set instance layout
  policy.set_instance_layout(layout);
  ASSERT_EQ(policy,
            legate::mapping::InstanceMappingPolicy{}
              .with_target(target)
              .with_allocation_policy(allocation)
              .with_instance_layout(layout));

  // Set ordering
  policy.set_ordering(dim_order);
  ASSERT_EQ(policy,
            legate::mapping::InstanceMappingPolicy{}
              .with_target(target)
              .with_allocation_policy(allocation)
              .with_instance_layout(layout)
              .with_ordering(dim_order));

  // Set exact
  policy.set_exact(true);
  ASSERT_EQ(policy,
            legate::mapping::InstanceMappingPolicy{}
              .with_target(target)
              .with_allocation_policy(allocation)
              .with_instance_layout(layout)
              .with_ordering(dim_order)
              .with_exact(true));
}

TEST_P(StoreTargetInput, Check)
{
  auto target = GetParam();

  const legate::mapping::InstanceMappingPolicy policy_a{};
  legate::mapping::InstanceMappingPolicy policy_b{};
  policy_b.set_target(target);
  check_subsume(policy_a, policy_b);
}

TEST_P(AllocationInput, Check)
{
  auto&& allocation = GetParam();

  const legate::mapping::InstanceMappingPolicy policy_a{};
  legate::mapping::InstanceMappingPolicy policy_b{};
  policy_b.set_allocation_policy(allocation);
  check_subsume(policy_a, policy_b);
}

TEST_P(InstanceInput, Check)
{
  auto&& instance = GetParam();

  const legate::mapping::InstanceMappingPolicy policy_a{};
  legate::mapping::InstanceMappingPolicy policy_b{};
  policy_b.set_instance_layout(instance);
  check_subsume(policy_a, policy_b);
}

TEST_P(DimOrderInput, Check)
{
  auto&& order = GetParam();

  const legate::mapping::InstanceMappingPolicy policy_a{};
  legate::mapping::InstanceMappingPolicy policy_b{};
  policy_b.set_ordering(order);
  check_subsume(policy_a, policy_b);
}

TEST_P(ExtractInput, Check)
{
  auto& [extract_a, extract_b] = GetParam();

  legate::mapping::InstanceMappingPolicy policy_a{};
  legate::mapping::InstanceMappingPolicy policy_b{};
  policy_a.set_exact(extract_a);
  policy_b.set_exact(extract_b);
  check_subsume(policy_a, policy_b);
}
}  // namespace unit
