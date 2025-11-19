/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/mapping/detail/instance_manager.h>

#include <legate.h>

#include <legate/data/detail/logical_store.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace instance_manager_unit {

using InstanceMgrTest         = DefaultFixture;
using InstanceMgrNegativeTest = DefaultFixture;

namespace {

// Helper function to create a test logical region
Legion::LogicalRegion create_test_region()
{
  auto runtime              = legate::Runtime::get_runtime();
  constexpr auto SHAPE_SIZE = 10;
  auto store                = runtime->create_store(legate::Shape{SHAPE_SIZE}, legate::int32());
  const auto& store_impl    = store.impl();
  auto region_field         = store_impl->get_region_field();

  LEGATE_CHECK(region_field != nullptr);
  return region_field->region();
}

}  // namespace

TEST_F(InstanceMgrTest, DefaultConstruction)
{
  const auto& instance_set = legate::mapping::detail::ReductionInstanceSet{};

  ASSERT_TRUE(instance_set.empty());
}

TEST_F(InstanceMgrTest, RecordInstance)
{
  legate::mapping::detail::ReductionInstanceSet instance_set;
  auto region               = create_test_region();
  const auto& instance      = Legion::Mapping::PhysicalInstance{};
  constexpr auto test_redop = legate::GlobalRedopID{1};
  const auto& policy        = legate::mapping::InstanceMappingPolicy{};

  // Initially empty
  ASSERT_TRUE(instance_set.empty());

  // Record instance
  instance_set.record_instance(test_redop, region, instance, policy);

  // Verify the instance was recorded
  ASSERT_FALSE(instance_set.empty());
}

TEST_F(InstanceMgrTest, RecordInstanceWithFind)
{
  legate::mapping::detail::ReductionInstanceSet instance_set;
  auto region               = create_test_region();
  const auto& instance      = Legion::Mapping::PhysicalInstance{};
  constexpr auto test_redop = legate::GlobalRedopID{1};
  const auto& policy        = legate::mapping::InstanceMappingPolicy{};

  instance_set.record_instance(test_redop, region, instance, policy);

  const auto& layout_constraints = Legion::LayoutConstraintSet{};
  std::optional<Legion::Mapping::PhysicalInstance> found;

  // find_instance should not crash with recorded instance
  ASSERT_NO_THROW({ found = instance_set.find_instance(test_redop, region, layout_constraints); })
    << "find_instance should not crash with recorded instance";
}

TEST_F(InstanceMgrTest, RecordInstanceReplacement)
{
  legate::mapping::detail::ReductionInstanceSet instance_set;
  auto region               = create_test_region();
  const auto& instance1     = Legion::Mapping::PhysicalInstance{};
  const auto& instance2     = Legion::Mapping::PhysicalInstance{};
  constexpr auto test_redop = legate::GlobalRedopID{1};
  const auto& policy        = legate::mapping::InstanceMappingPolicy{};

  // Record first instance
  instance_set.record_instance(test_redop, region, instance1, policy);
  ASSERT_FALSE(instance_set.empty());

  // Record to same region again (should replace, not add)
  instance_set.record_instance(test_redop, region, instance2, policy);
  ASSERT_FALSE(instance_set.empty());
}

TEST_F(InstanceMgrTest, RecordInstanceMultipleRegions)
{
  legate::mapping::detail::ReductionInstanceSet instance_set;
  auto region1              = create_test_region();
  auto region2              = create_test_region();
  const auto& instance      = Legion::Mapping::PhysicalInstance{};
  constexpr auto test_redop = legate::GlobalRedopID{1};
  const auto& policy        = legate::mapping::InstanceMappingPolicy{};

  // Record to first region
  instance_set.record_instance(test_redop, region1, instance, policy);
  ASSERT_FALSE(instance_set.empty());

  // Record to different region (should add)
  instance_set.record_instance(test_redop, region2, instance, policy);
  ASSERT_FALSE(instance_set.empty());
}

TEST_F(InstanceMgrNegativeTest, FindInstanceWithWrongRedop)
{
  legate::mapping::detail::ReductionInstanceSet instance_set;
  auto region                = create_test_region();
  const auto& instance       = Legion::Mapping::PhysicalInstance();
  constexpr auto test_redop  = legate::GlobalRedopID{1};
  constexpr auto wrong_redop = legate::GlobalRedopID{2};
  const auto& policy         = legate::mapping::InstanceMappingPolicy{};

  instance_set.record_instance(test_redop, region, instance, policy);

  const auto& layout_constraints = Legion::LayoutConstraintSet{};
  std::optional<Legion::Mapping::PhysicalInstance> not_found;

  ASSERT_NO_THROW({
    not_found = instance_set.find_instance(wrong_redop, region, layout_constraints);
  }) << "find_instance should not crash with wrong redop";

  ASSERT_FALSE(not_found.has_value()) << "Should not find instance when redop doesn't match";
}

TEST_F(InstanceMgrNegativeTest, FindInstanceWithWrongRegion)
{
  legate::mapping::detail::ReductionInstanceSet instance_set;
  auto region1              = create_test_region();
  auto region2              = create_test_region();
  const auto& instance      = Legion::Mapping::PhysicalInstance{};
  constexpr auto test_redop = legate::GlobalRedopID{1};
  const auto& policy        = legate::mapping::InstanceMappingPolicy{};

  instance_set.record_instance(test_redop, region1, instance, policy);

  const auto& layout_constraints = Legion::LayoutConstraintSet{};
  std::optional<Legion::Mapping::PhysicalInstance> not_found;

  ASSERT_NO_THROW({
    not_found = instance_set.find_instance(test_redop, region2, layout_constraints);
  }) << "find_instance should not crash with wrong region";

  ASSERT_FALSE(not_found.has_value()) << "Should not find instance when region doesn't match";
}

TEST_F(InstanceMgrTest, RecordMultipleInstances)
{
  legate::mapping::detail::ReductionInstanceSet instance_set;

  // Record multiple instances to different regions
  auto region1              = create_test_region();
  auto region2              = create_test_region();
  const auto& instance1     = Legion::Mapping::PhysicalInstance{};
  const auto& instance2     = Legion::Mapping::PhysicalInstance{};
  constexpr auto test_redop = legate::GlobalRedopID{1};
  const auto& policy        = legate::mapping::InstanceMappingPolicy{};

  instance_set.record_instance(test_redop, region1, instance1, policy);
  instance_set.record_instance(test_redop, region2, instance2, policy);
  ASSERT_FALSE(instance_set.empty());
}

TEST_F(InstanceMgrTest, RecordWithDifferentRedops)
{
  legate::mapping::detail::ReductionInstanceSet instance_set;

  // Use different reduction operators
  auto region1         = create_test_region();
  auto region2         = create_test_region();
  const auto& instance = Legion::Mapping::PhysicalInstance{};

  constexpr auto redop1 = legate::GlobalRedopID{1};
  constexpr auto redop2 = legate::GlobalRedopID{2};
  const auto& policy    = legate::mapping::InstanceMappingPolicy{};

  instance_set.record_instance(redop1, region1, instance, policy);
  instance_set.record_instance(redop2, region2, instance, policy);

  ASSERT_FALSE(instance_set.empty());
}

TEST_F(InstanceMgrTest, GetInstanceSizeWithEmptySet)
{
  const auto& instance_set = legate::mapping::detail::InstanceSet{};
  std::size_t size         = 0;

  ASSERT_NO_THROW({ size = instance_set.get_instance_size(); });
  ASSERT_EQ(size, 0);
}

}  // namespace instance_manager_unit
