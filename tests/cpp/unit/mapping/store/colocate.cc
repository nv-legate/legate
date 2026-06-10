/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/data/detail/logical_store.h>
#include <legate/mapping/detail/store.h>

#include <gtest/gtest.h>

#include <utilities/mock_mapper.h>
#include <utilities/utilities.h>

namespace mapping_store_test {

using legate::mapping::detail::FutureWrapper;
using legate::mapping::detail::RegionField;
using legate::mapping::detail::Store;

namespace {

using MappingStoreColocateTest = DefaultFixture;
using legate::test::MockMapperRuntime;

constexpr auto REGION_STORE_DIM = std::int32_t{1};

[[nodiscard]] Legion::RegionRequirement make_region_requirement(const Legion::LogicalRegion& region,
                                                                Legion::FieldID field_id)
{
  auto requirement = Legion::RegionRequirement{region, LEGION_READ_WRITE, LEGION_EXCLUSIVE, region};

  requirement.add_field(field_id);
  return requirement;
}

[[nodiscard]] RegionField make_region_field(const Legion::RegionRequirement& requirement,
                                            Legion::FieldID field_id)
{
  return RegionField{requirement, REGION_STORE_DIM, /*idx=*/0, field_id, /*unbound=*/false};
}

}  // namespace

TEST_F(MappingStoreColocateTest, FutureStores)
{
  const legate::InternalSharedPtr<legate::detail::Type> type1{legate::int32().impl()};
  const legate::InternalSharedPtr<legate::detail::Type> type2{legate::float32().impl()};
  const Legion::Domain domain1{Legion::Rect<1>{0, 9}};
  const Legion::Domain domain2{Legion::Rect<1>{0, 3}};
  const FutureWrapper future1{/*idx=*/0, domain1};
  const FutureWrapper future2{/*idx=*/1, domain2};
  const Store store1{/*dim=*/1, type1, future1};
  const Store store2{/*dim=*/1, type2, future2};

  // Future stores cannot colocate with anything
  ASSERT_FALSE(store1.can_colocate_with(store2));
  ASSERT_FALSE(store2.can_colocate_with(store1));
}

TEST_F(MappingStoreColocateTest, FutureWithItself)
{
  const legate::InternalSharedPtr<legate::detail::Type> type{legate::int32().impl()};
  const Legion::Domain domain{Legion::Rect<1>{0, 9}};
  const FutureWrapper future{/*idx=*/0, domain};
  const Store store{/*dim=*/1, type, future};

  // Even with itself, future cannot colocate
  ASSERT_FALSE(store.can_colocate_with(store));
}

TEST_F(MappingStoreColocateTest, FutureStoresDifferentDims)
{
  const legate::InternalSharedPtr<legate::detail::Type> type{legate::int32().impl()};
  const Legion::Domain domain1d{Legion::Rect<1>{0, 9}};
  const Legion::Domain domain2d{Legion::Rect<2>{{0, 0}, {4, 4}}};
  const FutureWrapper future1d{/*idx=*/0, domain1d};
  const FutureWrapper future2d{/*idx=*/1, domain2d};
  const Store store1d{/*dim=*/1, type, future1d};
  const Store store2d{/*dim=*/2, type, future2d};

  // Future stores of different dimensions still cannot colocate
  ASSERT_FALSE(store1d.can_colocate_with(store2d));
  ASSERT_FALSE(store2d.can_colocate_with(store1d));
}

TEST_F(MappingStoreColocateTest, FutureStoresSameIndexDifferentDomain)
{
  const legate::InternalSharedPtr<legate::detail::Type> type{legate::int32().impl()};
  const Legion::Domain domain1{Legion::Rect<1>{0, 9}};
  const Legion::Domain domain2{Legion::Rect<1>{0, 19}};

  // Same future index but different domains
  const FutureWrapper future1{/*idx=*/0, domain1};
  const FutureWrapper future2{/*idx=*/0, domain2};
  const Store store1{/*dim=*/1, type, future1};
  const Store store2{/*dim=*/1, type, future2};

  // Future stores cannot colocate even with same index
  ASSERT_FALSE(store1.can_colocate_with(store2));
  ASSERT_FALSE(store2.can_colocate_with(store1));
}

TEST_F(MappingStoreColocateTest, RegionStoreWithFutureStore)
{
  auto* const runtime = legate::Runtime::get_runtime();
  auto logical_store  = runtime->create_store(legate::Shape{4}, legate::int32());
  auto region_field   = logical_store.impl()->get_region_field();
  auto requirement    = make_region_requirement(region_field->region(), region_field->field_id());
  const auto field    = make_region_field(requirement, region_field->field_id());
  const auto type     = legate::InternalSharedPtr<legate::detail::Type>{legate::int32().impl()};
  MockMapperRuntime mapper_runtime;
  const auto context = Legion::Mapping::MapperContext{};
  const Store region_store{
    mapper_runtime, context, REGION_STORE_DIM, type, legate::GlobalRedopID{0}, field};
  const auto domain = Legion::Domain{Legion::Rect<1>{0, 3}};
  const auto future = FutureWrapper{/*idx=*/0, domain};
  const Store future_store{REGION_STORE_DIM, type, future};

  ASSERT_FALSE(region_store.is_future());
  ASSERT_TRUE(future_store.is_future());
  ASSERT_FALSE(region_store.can_colocate_with(future_store));
}

TEST_F(MappingStoreColocateTest, RegularStoreWithReductionStore)
{
  auto* const runtime = legate::Runtime::get_runtime();
  auto logical_store  = runtime->create_store(legate::Shape{4}, legate::int32());
  auto region_field   = logical_store.impl()->get_region_field();
  auto requirement    = make_region_requirement(region_field->region(), region_field->field_id());
  const auto field    = make_region_field(requirement, region_field->field_id());
  const auto type     = legate::InternalSharedPtr<legate::detail::Type>{legate::int32().impl()};
  MockMapperRuntime mapper_runtime;
  const auto context = Legion::Mapping::MapperContext{};
  const Store regular_store{
    mapper_runtime, context, REGION_STORE_DIM, type, legate::GlobalRedopID{0}, field};
  const Store reduction_store{
    mapper_runtime, context, REGION_STORE_DIM, type, legate::GlobalRedopID{1}, field};

  ASSERT_FALSE(regular_store.is_reduction());
  ASSERT_TRUE(reduction_store.is_reduction());
  ASSERT_FALSE(regular_store.can_colocate_with(reduction_store));
}

TEST_F(MappingStoreColocateTest, RegionFieldsCanColocate)
{
  auto* const runtime = legate::Runtime::get_runtime();
  auto store          = runtime->create_store(legate::Shape{4}, legate::int32());
  auto region_field   = store.impl()->get_region_field();
  constexpr auto dim  = std::int32_t{1};
  auto requirement    = make_region_requirement(region_field->region(), region_field->field_id());
  const auto field =
    RegionField{requirement, dim, /*idx=*/0, region_field->field_id(), /*unbound=*/false};
  const auto same =
    RegionField{requirement, dim, /*idx=*/1, region_field->field_id(), /*unbound=*/false};

  ASSERT_TRUE(field.can_colocate_with(same));
}

TEST_F(MappingStoreColocateTest, RegionFieldIndexSpace)
{
  auto* const runtime = legate::Runtime::get_runtime();
  auto store          = runtime->create_store(legate::Shape{4}, legate::int32());
  auto region_field   = store.impl()->get_region_field();
  constexpr auto dim  = std::int32_t{1};
  auto requirement    = make_region_requirement(region_field->region(), region_field->field_id());
  const auto field =
    RegionField{requirement, dim, /*idx=*/0, region_field->field_id(), /*unbound=*/false};

  ASSERT_EQ(field.get_index_space(), region_field->region().get_index_space());
}

TEST_F(MappingStoreColocateTest, RegionFieldsDifferentTreesCannotColocate)
{
  auto* const runtime    = legate::Runtime::get_runtime();
  auto first_store       = runtime->create_store(legate::Shape{4}, legate::int32());
  auto second_store      = runtime->create_store(legate::Shape{5}, legate::int32());
  auto first_field       = first_store.impl()->get_region_field();
  auto second_field      = second_store.impl()->get_region_field();
  constexpr auto dim     = std::int32_t{1};
  auto first_requirement = make_region_requirement(first_field->region(), first_field->field_id());
  auto second_requirement =
    make_region_requirement(second_field->region(), first_field->field_id());
  const auto field =
    RegionField{first_requirement, dim, /*idx=*/0, first_field->field_id(), /*unbound=*/false};
  const auto other =
    RegionField{second_requirement, dim, /*idx=*/1, first_field->field_id(), /*unbound=*/false};

  ASSERT_NE(first_field->region().get_tree_id(), second_field->region().get_tree_id());
  ASSERT_FALSE(field.can_colocate_with(other));
}

TEST_F(MappingStoreColocateTest, RegionFieldsDifferentFieldsCannotColocate)
{
  auto* const runtime       = legate::Runtime::get_runtime();
  auto store                = runtime->create_store(legate::Shape{4}, legate::int32());
  auto region_field         = store.impl()->get_region_field();
  constexpr auto dim        = std::int32_t{1};
  const auto other_field_id = static_cast<Legion::FieldID>(region_field->field_id() + 1);
  auto requirement = make_region_requirement(region_field->region(), region_field->field_id());
  auto other_requirement = make_region_requirement(region_field->region(), other_field_id);
  const auto field =
    RegionField{requirement, dim, /*idx=*/0, region_field->field_id(), /*unbound=*/false};
  const auto other =
    RegionField{other_requirement, dim, /*idx=*/1, other_field_id, /*unbound=*/false};

  ASSERT_FALSE(field.can_colocate_with(other));
}

TEST_F(MappingStoreColocateTest, RegionFieldsDifferentDimsCannotColocate)
{
  auto* const runtime      = legate::Runtime::get_runtime();
  auto store               = runtime->create_store(legate::Shape{4}, legate::int32());
  auto region_field        = store.impl()->get_region_field();
  constexpr auto dim       = std::int32_t{1};
  constexpr auto other_dim = std::int32_t{2};
  auto requirement = make_region_requirement(region_field->region(), region_field->field_id());
  const auto field =
    RegionField{requirement, dim, /*idx=*/0, region_field->field_id(), /*unbound=*/false};
  const auto other =
    RegionField{requirement, other_dim, /*idx=*/1, region_field->field_id(), /*unbound=*/false};

  ASSERT_FALSE(field.can_colocate_with(other));
}

}  // namespace mapping_store_test
