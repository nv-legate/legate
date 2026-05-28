/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/mapping/detail/instance_manager.h>

#include <legate.h>

#include <legate/data/detail/logical_store.h>

#include <gtest/gtest.h>

#include <sstream>
#include <utilities/utilities.h>

namespace instance_manager_unit {

using InstanceMgrTest         = DefaultFixture;
using InstanceMgrNegativeTest = DefaultFixture;

namespace {

using InstanceSet = legate::mapping::detail::InstanceSet;

constexpr std::uint64_t DEFAULT_SHAPE_SIZE     = 10;
constexpr std::uint64_t OVERLAPPING_SHAPE_SIZE = 15;

// Helper function to create a test logical region
Legion::LogicalRegion create_test_region(std::uint64_t shape_size = DEFAULT_SHAPE_SIZE)
{
  auto runtime           = legate::Runtime::get_runtime();
  auto store             = runtime->create_store(legate::Shape{shape_size}, legate::int32());
  const auto& store_impl = store.impl();
  auto region_field      = store_impl->get_region_field();

  LEGATE_CHECK(region_field != nullptr);
  return region_field->region();
}

struct InstanceSetEraseScenario {
  InstanceSet instance_set{};
  Legion::Mapping::PhysicalInstance instance{};
  Legion::Mapping::PhysicalInstance exact_instance{};
};

InstanceSetEraseScenario make_instance_set_erase_scenario()
{
  InstanceSetEraseScenario scenario;
  auto region1                         = create_test_region(/*shape_size=*/DEFAULT_SHAPE_SIZE);
  auto region2                         = create_test_region(/*shape_size=*/OVERLAPPING_SHAPE_SIZE);
  constexpr legate::coord_t low        = 0;
  constexpr legate::coord_t mid        = 5;
  constexpr legate::coord_t high       = 9;
  constexpr legate::coord_t upper_high = 14;
  const auto lower_domain              = legate::Domain{legate::Rect<1>{low, high}};
  const auto upper_domain              = legate::Domain{legate::Rect<1>{mid, upper_high}};
  const auto& policy                   = legate::mapping::InstanceMappingPolicy{};
  auto exact_policy                    = policy;
  exact_policy.exact                   = true;
  scenario.exact_instance              = Legion::Mapping::PhysicalInstance::get_virtual_instance();

  const auto initial_group =
    scenario.instance_set.find_or_create_region_group(region1, lower_domain, /*exact=*/false);
  scenario.instance_set.record_pending_instance_creation(initial_group);
  scenario.instance_set.record_instance(region1, initial_group, scenario.instance, policy);

  const auto overlapping_group =
    scenario.instance_set.find_or_create_region_group(region2, upper_domain, /*exact=*/false);
  scenario.instance_set.record_pending_instance_creation(overlapping_group);
  scenario.instance_set.record_instance(region2, overlapping_group, scenario.instance, policy);

  const auto exact_group =
    scenario.instance_set.find_or_create_region_group(region1, lower_domain, /*exact=*/true);
  scenario.instance_set.record_pending_instance_creation(exact_group);
  scenario.instance_set.record_instance(
    region1, exact_group, scenario.exact_instance, exact_policy);

  return scenario;
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

TEST_F(InstanceMgrTest, RecordInstanceReplacesSameRegionWhenPolicyOrRedopDiffers)
{
  legate::mapping::detail::ReductionInstanceSet instance_set;
  auto region                  = create_test_region();
  const auto& instance         = Legion::Mapping::PhysicalInstance{};
  constexpr auto first_redop   = legate::GlobalRedopID{1};
  constexpr auto updated_redop = legate::GlobalRedopID{2};
  const auto& policy           = legate::mapping::InstanceMappingPolicy{};
  auto exact_policy            = policy;
  exact_policy.exact           = true;

  instance_set.record_instance(first_redop, region, instance, policy);
  instance_set.record_instance(updated_redop, region, instance, policy);
  instance_set.record_instance(updated_redop, region, instance, exact_policy);

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
  const auto& instance       = Legion::Mapping::PhysicalInstance{};
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
  auto region1              = create_test_region(/*shape_size=*/DEFAULT_SHAPE_SIZE);
  auto region2              = create_test_region(/*shape_size=*/OVERLAPPING_SHAPE_SIZE);
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

TEST_F(InstanceMgrTest, ReductionInstanceSetEraseMatchingInstance)
{
  legate::mapping::detail::ReductionInstanceSet instance_set;
  auto region1               = create_test_region(/*shape_size=*/DEFAULT_SHAPE_SIZE);
  auto region2               = create_test_region(/*shape_size=*/OVERLAPPING_SHAPE_SIZE);
  const auto& instance       = Legion::Mapping::PhysicalInstance{};
  const auto& other_instance = Legion::Mapping::PhysicalInstance::get_virtual_instance();
  constexpr auto test_redop  = legate::GlobalRedopID{1};
  const auto& policy         = legate::mapping::InstanceMappingPolicy{};

  instance_set.record_instance(test_redop, region1, instance, policy);
  instance_set.record_instance(test_redop, region2, other_instance, policy);

  ASSERT_TRUE(instance_set.erase(instance));
  ASSERT_FALSE(instance_set.empty());
  ASSERT_TRUE(instance_set.erase(other_instance));
  ASSERT_TRUE(instance_set.empty());
}

TEST_F(InstanceMgrTest, ReductionInstanceSetEraseMissingInstance)
{
  legate::mapping::detail::ReductionInstanceSet instance_set;
  auto region                  = create_test_region();
  const auto& instance         = Legion::Mapping::PhysicalInstance{};
  const auto& missing_instance = Legion::Mapping::PhysicalInstance::get_virtual_instance();
  constexpr auto test_redop    = legate::GlobalRedopID{1};
  const auto& policy           = legate::mapping::InstanceMappingPolicy{};

  ASSERT_FALSE(instance_set.erase(missing_instance));

  instance_set.record_instance(test_redop, region, instance, policy);

  ASSERT_FALSE(instance_set.erase(missing_instance));
  ASSERT_FALSE(instance_set.empty());
}

TEST_F(InstanceMgrTest, ReductionInstanceManagerEraseKeepsNonemptyInstanceSet)
{
  legate::mapping::detail::ReductionInstanceManager instance_manager;
  auto region1                       = create_test_region(/*shape_size=*/DEFAULT_SHAPE_SIZE);
  auto region2                       = create_test_region(/*shape_size=*/OVERLAPPING_SHAPE_SIZE);
  const auto& instance               = Legion::Mapping::PhysicalInstance{};
  const auto& other_instance         = Legion::Mapping::PhysicalInstance::get_virtual_instance();
  constexpr auto test_redop          = legate::GlobalRedopID{1};
  constexpr Legion::FieldID field_id = 1;
  const auto& policy                 = legate::mapping::InstanceMappingPolicy{};

  instance_manager.record_instance(test_redop, region1, field_id, instance, policy);
  instance_manager.record_instance(test_redop, region2, field_id, other_instance, policy);

  ASSERT_TRUE(instance_manager.erase(instance));
  ASSERT_TRUE(instance_manager.erase(other_instance));
}

TEST_F(InstanceMgrTest, InstanceManagerEraseSkipsMismatchedKeys)
{
  legate::mapping::detail::InstanceManager instance_manager;
  auto region1                       = create_test_region(/*shape_size=*/DEFAULT_SHAPE_SIZE);
  auto region2                       = create_test_region(/*shape_size=*/OVERLAPPING_SHAPE_SIZE);
  constexpr legate::coord_t low      = 0;
  constexpr legate::coord_t high     = 9;
  constexpr Legion::FieldID field_id = 1;
  const auto domain                  = legate::Domain{legate::Rect<1>{low, high}};
  const auto& instance               = Legion::Mapping::PhysicalInstance{};
  const auto memory                  = instance.get_location();
  const auto other_memory            = Legion::Memory{memory.id + 1};

  const auto other_memory_group =
    instance_manager.find_region_group(region1, domain, field_id, other_memory, /*exact=*/false);
  const auto other_tree_group =
    instance_manager.find_region_group(region2, domain, field_id, memory, /*exact=*/false);

  ASSERT_FALSE(instance_manager.erase(instance));

  instance_manager.remove_pending_instance(region1, other_memory_group, field_id, other_memory);
  instance_manager.remove_pending_instance(region2, other_tree_group, field_id, memory);
}

TEST_F(InstanceMgrTest, InstanceManagerRecordInstanceReusesInstanceSet)
{
  legate::mapping::detail::InstanceManager instance_manager;
  const auto region                  = Legion::LogicalRegion::NO_REGION;
  constexpr legate::coord_t low      = 0;
  constexpr legate::coord_t high     = 9;
  constexpr Legion::FieldID field_id = 1;
  const auto domain                  = legate::Domain{legate::Rect<1>{low, high}};
  const auto& instance               = Legion::Mapping::PhysicalInstance{};
  const auto& replacement_instance   = Legion::Mapping::PhysicalInstance::get_virtual_instance();
  const auto memory                  = instance.get_location();
  const auto& policy                 = legate::mapping::InstanceMappingPolicy{};

  const auto group =
    instance_manager.find_region_group(region, domain, field_id, memory, /*exact=*/false);
  instance_manager.record_instance(region, group, field_id, instance, policy);

  const auto existing_group =
    instance_manager.find_region_group(region, domain, field_id, memory, /*exact=*/false);
  instance_manager.record_instance(region, existing_group, field_id, replacement_instance, policy);

  const auto pending_group =
    instance_manager.find_region_group(region, domain, field_id, memory, /*exact=*/false);
  instance_manager.remove_pending_instance(region, pending_group, field_id, memory);

  ASSERT_TRUE(instance_manager.erase(replacement_instance));
}

TEST_F(InstanceMgrTest, FieldMemInfoEqualityComparesAllFields)
{
  using FieldMemInfo = legate::mapping::detail::BaseInstanceManager::FieldMemInfo;

  constexpr Legion::RegionTreeID base_tid  = 1;
  constexpr Legion::RegionTreeID other_tid = 2;
  constexpr Legion::FieldID base_fid       = 1;
  constexpr Legion::FieldID other_fid      = 2;
  const auto base_memory                   = Legion::Memory{1};
  const auto other_memory                  = Legion::Memory{2};
  const auto base_info                     = FieldMemInfo{base_tid, base_fid, base_memory};

  ASSERT_EQ(base_info, (FieldMemInfo{base_tid, base_fid, base_memory}));
  ASSERT_FALSE((base_info == FieldMemInfo{other_tid, base_fid, base_memory}));
  ASSERT_FALSE((base_info == FieldMemInfo{base_tid, other_fid, base_memory}));
  ASSERT_FALSE((base_info == FieldMemInfo{base_tid, base_fid, other_memory}));
}

TEST_F(InstanceMgrTest, GetInstanceSizeWithEmptySet)
{
  const auto& instance_set = legate::mapping::detail::InstanceSet{};
  std::size_t size         = 0;

  ASSERT_NO_THROW({ size = instance_set.get_instance_size(); });
  ASSERT_EQ(size, 0);
}

TEST_F(InstanceMgrTest, GetInstanceSizeWithMultipleInstances)
{
  legate::mapping::detail::InstanceSet instance_set;
  auto region1                    = create_test_region(/*shape_size=*/DEFAULT_SHAPE_SIZE);
  auto region2                    = create_test_region(/*shape_size=*/OVERLAPPING_SHAPE_SIZE);
  constexpr legate::coord_t low   = 0;
  constexpr legate::coord_t high1 = 9;
  constexpr legate::coord_t high2 = 14;
  const auto domain1              = legate::Domain{legate::Rect<1>{low, high1}};
  const auto domain2              = legate::Domain{legate::Rect<1>{low, high2}};
  const auto& instance1           = Legion::Mapping::PhysicalInstance{};
  const auto& instance2           = Legion::Mapping::PhysicalInstance::get_virtual_instance();
  const auto& policy              = legate::mapping::InstanceMappingPolicy{};

  const auto group1 = instance_set.find_or_create_region_group(region1, domain1, /*exact=*/true);
  instance_set.record_pending_instance_creation(group1);
  instance_set.record_instance(region1, group1, instance1, policy);

  const auto group2 = instance_set.find_or_create_region_group(region2, domain2, /*exact=*/true);
  instance_set.record_pending_instance_creation(group2);
  instance_set.record_instance(region2, group2, instance2, policy);

  ASSERT_EQ(instance_set.get_instance_size(), 0);
}

TEST_F(InstanceMgrTest, RegionGroupToString)
{
  auto region                    = create_test_region();
  constexpr legate::coord_t low  = 0;
  constexpr legate::coord_t high = 9;
  const auto domain              = legate::Domain{legate::Rect<1>{low, high}};
  const auto region_group =
    legate::mapping::detail::RegionGroup{std::set<Legion::LogicalRegion>{region}, domain};
  std::stringstream region_stream;
  region_stream << region;

  std::stringstream expected;
  expected << "RegionGroup(" << domain << ": {" << region_stream.str() << ",})";

  std::stringstream result;
  result << region_group;

  ASSERT_EQ(result.str(), expected.str());
}

TEST_F(InstanceMgrTest, InstanceSetEmptyTracksPendingInstances)
{
  legate::mapping::detail::InstanceSet instance_set;
  auto region                    = create_test_region();
  constexpr legate::coord_t low  = 0;
  constexpr legate::coord_t high = 9;
  const auto domain              = legate::Domain{legate::Rect<1>{low, high}};
  const auto group = instance_set.find_or_create_region_group(region, domain, /*exact=*/false);

  instance_set.record_pending_instance_creation(group);
  ASSERT_FALSE(instance_set.empty());
  instance_set.remove_pending_instance(group);

  ASSERT_TRUE(instance_set.empty());
}

TEST_F(InstanceMgrTest, FindRegionGroupCombinesPendingGroup)
{
  legate::mapping::detail::InstanceSet instance_set;
  auto region1                         = create_test_region(/*shape_size=*/DEFAULT_SHAPE_SIZE);
  auto region2                         = create_test_region(/*shape_size=*/OVERLAPPING_SHAPE_SIZE);
  constexpr legate::coord_t low        = 0;
  constexpr legate::coord_t mid        = 5;
  constexpr legate::coord_t high       = 9;
  constexpr legate::coord_t upper_high = 14;
  const auto lower_domain              = legate::Domain{legate::Rect<1>{low, high}};
  const auto upper_domain              = legate::Domain{legate::Rect<1>{mid, upper_high}};
  const auto pending_group =
    instance_set.find_or_create_region_group(region1, lower_domain, /*exact=*/false);
  instance_set.record_pending_instance_creation(pending_group);

  const auto combined_group =
    instance_set.find_or_create_region_group(region2, upper_domain, /*exact=*/false);

  ASSERT_EQ(combined_group->regions.size(), 2);
  ASSERT_EQ(combined_group->regions.count(region1), 1);
  ASSERT_EQ(combined_group->regions.count(region2), 1);

  instance_set.remove_pending_instance(pending_group);
}

TEST_F(InstanceMgrTest, FindRegionGroupSkipsSelfComparison)
{
  legate::mapping::detail::InstanceSet instance_set;
  auto region                    = create_test_region();
  constexpr legate::coord_t low  = 0;
  constexpr legate::coord_t high = 9;
  const auto domain              = legate::Domain{legate::Rect<1>{low, high}};
  const auto group     = instance_set.find_or_create_region_group(region, domain, /*exact=*/false);
  const auto& instance = Legion::Mapping::PhysicalInstance{};
  const auto& policy   = legate::mapping::InstanceMappingPolicy{};

  instance_set.record_pending_instance_creation(group);
  // record_instance() consumes the first pending creation for this group.
  instance_set.record_instance(region, group, instance, policy);
  instance_set.record_pending_instance_creation(group);

  const auto same_group = instance_set.find_or_create_region_group(region, domain, /*exact=*/false);

  ASSERT_EQ(same_group.get(), group.get());

  instance_set.remove_pending_instance(group);
}

TEST_F(InstanceMgrTest, EraseSkipsRegionsMappedToDifferentGroup)
{
  auto scenario = make_instance_set_erase_scenario();

  ASSERT_TRUE(scenario.instance_set.erase(scenario.instance));
  ASSERT_FALSE(scenario.instance_set.empty());
  ASSERT_TRUE(scenario.instance_set.erase(scenario.exact_instance));
  ASSERT_TRUE(scenario.instance_set.empty());
}

TEST_F(InstanceMgrTest, EraseSkipsRegionsAlreadyRemoved)
{
  auto scenario = make_instance_set_erase_scenario();

  ASSERT_TRUE(scenario.instance_set.erase(scenario.exact_instance));
  ASSERT_FALSE(scenario.instance_set.empty());
  ASSERT_TRUE(scenario.instance_set.erase(scenario.instance));
  ASSERT_TRUE(scenario.instance_set.empty());
}

}  // namespace instance_manager_unit
