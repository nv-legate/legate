/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/logical_region_field.h>

#include <legate.h>

#include <legate/data/detail/logical_store.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/utilities/detail/tuple.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace logical_region_field_test {

using LogicalRegionFieldUnit = DefaultFixture;

using LogicalRegionFieldDeathTest = LogicalRegionFieldUnit;

class CreateLogicalRegionFieldTest
  : public LogicalRegionFieldUnit,
    public ::testing::WithParamInterface<std::tuple<legate::Shape, legate::Type, std::uint32_t>> {};

INSTANTIATE_TEST_SUITE_P(LogicalRegionFieldUnit,
                         CreateLogicalRegionFieldTest,
                         ::testing::Combine(::testing::Values(legate::Shape{1},
                                                              legate::Shape{7, 8, 9, 10}),
                                            ::testing::Values(legate::bool_(),
                                                              legate::int8(),
                                                              legate::uint64(),
                                                              legate::float16(),
                                                              legate::complex64()),
                                            ::testing::Values(0, 20, 1000)));

TEST_P(CreateLogicalRegionFieldTest, Create)
{
  auto [shape, element_type, field_id] = GetParam();
  auto runtime                         = legate::Runtime::get_runtime();
  auto store                           = runtime->create_store(shape, element_type);
  auto region_field                    = store.impl()->get_region_field();

  ASSERT_EQ(region_field->dim(), shape.ndim());
  ASSERT_TRUE(region_field->region() != Legion::LogicalRegion::NO_REGION);
  ASSERT_NE(region_field->field_id(), static_cast<Legion::FieldID>(field_id));
  ASSERT_EQ(region_field->parent(), std::nullopt);
  ASSERT_EQ(&region_field->get_root(), region_field.get());
  ASSERT_FALSE(region_field->is_mapped());

  auto domain = legate::detail::to_domain(shape.extents());

  ASSERT_EQ(region_field->domain(), domain);
}

TEST_F(LogicalRegionFieldUnit, Parent)
{
  auto runtime             = legate::Runtime::get_runtime();
  auto store               = runtime->create_store(legate::Shape{3}, legate::int32());
  auto parent_region_field = store.impl()->get_region_field();

  auto shape = legate::make_internal_shared<legate::detail::Shape>(LEGATE_MAX_DIM);
  constexpr std::uint32_t filed_size = 100;
  auto field_id                      = parent_region_field->field_id() + 1;
  auto child_region_field            = legate::detail::LogicalRegionField{
    shape, filed_size, parent_region_field->region(), field_id, parent_region_field};

  ASSERT_EQ(child_region_field.region(), parent_region_field->region());
  ASSERT_EQ(child_region_field.field_id(), field_id);
  ASSERT_TRUE(child_region_field.parent().has_value());
  ASSERT_EQ(&child_region_field.get_root(), parent_region_field.get());
}

TEST_F(LogicalRegionFieldUnit, Unmap)
{
  auto runtime      = legate::Runtime::get_runtime();
  auto store        = runtime->create_store(legate::Shape{1}, legate::int64());
  auto region_field = store.impl()->get_region_field();
  auto map          = region_field->map(legate::mapping::StoreTarget::SYSMEM);

  ASSERT_TRUE(region_field->is_mapped());
  region_field->unmap();
  ASSERT_FALSE(region_field->is_mapped());
}

TEST_F(LogicalRegionFieldUnit, UnmapParent)
{
  auto runtime             = legate::Runtime::get_runtime();
  auto type                = legate::uint64();
  auto store               = runtime->create_store(legate::Shape{1}, type);
  auto parent_region_field = store.impl()->get_region_field();
  auto child_shape         = legate::make_internal_shared<legate::detail::Shape>(1);
  auto field_size          = type.size();
  auto child_region_field  = legate::make_internal_shared<legate::detail::LogicalRegionField>(
    child_shape,
    field_size,
    parent_region_field->region(),
    parent_region_field->field_id(),
    parent_region_field);

  ASSERT_FALSE(parent_region_field->is_mapped());
  ASSERT_FALSE(child_region_field->is_mapped());

  auto parent_map = parent_region_field->map(legate::mapping::StoreTarget::SYSMEM);
  ASSERT_TRUE(parent_region_field->is_mapped());
  ASSERT_TRUE(child_region_field->is_mapped());

  // unmap parent
  parent_region_field->unmap();
  ASSERT_FALSE(parent_region_field->is_mapped());
  ASSERT_FALSE(child_region_field->is_mapped());
}

TEST_F(LogicalRegionFieldUnit, UnmapThroughChild)
{
  auto runtime             = legate::Runtime::get_runtime();
  auto type                = legate::uint32();
  auto store               = runtime->create_store(legate::Shape{2}, type);
  auto parent_region_field = store.impl()->get_region_field();
  auto parent_map          = parent_region_field->map(legate::mapping::StoreTarget::SYSMEM);

  ASSERT_TRUE(parent_region_field->is_mapped());

  auto child_shape        = legate::make_internal_shared<legate::detail::Shape>(1);
  auto field_size         = type.size();
  auto child_region_field = legate::make_internal_shared<legate::detail::LogicalRegionField>(
    child_shape,
    field_size,
    parent_region_field->region(),
    parent_region_field->field_id(),
    parent_region_field);
  auto child_map = child_region_field->map(legate::mapping::StoreTarget::SYSMEM);

  ASSERT_TRUE(child_region_field->is_mapped());
  ASSERT_TRUE(child_map.valid());
  ASSERT_EQ(child_map.dim(), child_region_field->dim());
  ASSERT_EQ(child_map.domain(), child_region_field->domain());
  ASSERT_EQ(child_map.target(), legate::mapping::StoreTarget::SYSMEM);
  ASSERT_EQ(child_map.get_field_id(), child_region_field->field_id());

  ASSERT_NO_THROW(child_region_field->allow_out_of_order_destruction());
  ASSERT_NO_THROW(child_region_field->unmap());
  ASSERT_FALSE(child_region_field->is_mapped());
  ASSERT_FALSE(parent_region_field->is_mapped());
}

TEST_F(LogicalRegionFieldUnit, AttachWithPhysicalRegion)
{
  auto runtime      = legate::Runtime::get_runtime();
  auto store        = runtime->create_store(legate::Shape{2}, legate::int64());
  auto region_field = store.impl()->get_region_field();

  auto test_buffer = std::vector<std::int64_t>{1, 2};
  auto buffer_size = test_buffer.size() * sizeof(test_buffer.front());

  auto realm_resource = std::make_unique<Realm::ExternalMemoryResource>(
    reinterpret_cast<std::uintptr_t>(test_buffer.data()), buffer_size, false /* read_only */
  );
  auto allocation = legate::make_internal_shared<legate::detail::ExternalAllocation>(
    false /* read_only */,
    legate::mapping::StoreTarget::SYSMEM,
    test_buffer.data(),
    buffer_size,
    std::move(realm_resource));
  auto launcher = Legion::IndexAttachLauncher{legion_external_resource_t::LEGION_EXTERNAL_INSTANCE,
                                              region_field->region()};

  launcher.add_external_resource(region_field->region(), allocation->resource());
  launcher.constraints.field_constraint.field_set  = {region_field->field_id()};
  launcher.constraints.field_constraint.contiguous = false;
  launcher.constraints.field_constraint.inorder    = false;
  launcher.constraints.ordering_constraint.ordering.clear();
  launcher.constraints.ordering_constraint.ordering.push_back(DIM_X);
  launcher.constraints.ordering_constraint.ordering.push_back(DIM_F);
  launcher.privilege_fields.insert(region_field->field_id());

  auto external_resources = Legion::Runtime::get_runtime()->attach_external_resources(
    Legion::Runtime::get_context(), launcher);
  std::vector<legate::InternalSharedPtr<legate::detail::ExternalAllocation>> allocations{
    allocation};

  ASSERT_NO_THROW(region_field->mark_pending_attach());
  ASSERT_NO_THROW(region_field->attach(std::move(external_resources), std::move(allocations)));
  ASSERT_TRUE(region_field->is_mapped());
  ASSERT_NO_THROW(region_field->allow_out_of_order_destruction());
}

TEST_F(LogicalRegionFieldDeathTest, AttachAfterMap)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto store         = runtime->create_store(legate::Shape{9}, legate::uint64());
  auto region_field  = store.impl()->get_region_field();
  auto mapped_region = region_field->map(legate::mapping::StoreTarget::SYSMEM);
  ASSERT_TRUE(mapped_region.valid());
  auto test_buffer    = std::vector<std::int64_t>{2, 1};
  auto buffer_size    = test_buffer.size() * sizeof(test_buffer.front());
  auto realm_resource = std::make_unique<Realm::ExternalMemoryResource>(
    reinterpret_cast<std::uintptr_t>(test_buffer.data()), buffer_size, false /* read_only */
  );
  auto allocation = legate::make_internal_shared<legate::detail::ExternalAllocation>(
    false /* read_only */,
    legate::mapping::StoreTarget::SYSMEM,
    test_buffer.data(),
    buffer_size,
    std::move(realm_resource));
  const auto& physical_region = mapped_region.get_physical_region();

  // failed with attach after mapping is already done
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    ASSERT_EXIT(region_field->attach(physical_region, allocation),
                ::testing::KilledBySignal(SIGABRT),
                ::testing::HasSubstr("!physical_state_->physical_region().exists()"));
  }
}

TEST_F(LogicalRegionFieldUnit, AttachWithExternalResources)
{
  auto runtime        = legate::Runtime::get_runtime();
  auto store          = runtime->create_store(legate::Shape{3, 1}, legate::uint8());
  auto region_field   = store.impl()->get_region_field();
  auto test_buffer    = std::vector<std::uint8_t>{1, 2, 3};
  auto buffer_size    = test_buffer.size() * sizeof(test_buffer.front());
  auto realm_resource = std::make_unique<Realm::ExternalMemoryResource>(
    reinterpret_cast<std::uintptr_t>(test_buffer.data()), buffer_size, false);
  auto allocation = legate::make_internal_shared<legate::detail::ExternalAllocation>(
    false /* read_only */,
    legate::mapping::StoreTarget::SYSMEM,
    test_buffer.data(),
    buffer_size,
    std::move(realm_resource));
  auto launcher = Legion::IndexAttachLauncher{legion_external_resource_t::LEGION_EXTERNAL_INSTANCE,
                                              region_field->region()};

  launcher.add_external_resource(region_field->region(), allocation->resource());
  launcher.privilege_fields.insert(region_field->field_id());
  launcher.constraints.field_constraint.field_set  = {region_field->field_id()};
  launcher.constraints.field_constraint.contiguous = false;
  launcher.constraints.field_constraint.inorder    = false;
  launcher.constraints.ordering_constraint.ordering.clear();
  launcher.constraints.ordering_constraint.ordering.push_back(DIM_X);
  launcher.constraints.ordering_constraint.ordering.push_back(DIM_Y);
  launcher.constraints.ordering_constraint.ordering.push_back(DIM_F);

  auto external_resources = Legion::Runtime::get_runtime()->attach_external_resources(
    Legion::Runtime::get_context(), launcher);
  std::vector<legate::InternalSharedPtr<legate::detail::ExternalAllocation>> allocations{
    allocation};

  ASSERT_NO_THROW(region_field->mark_pending_attach());
  ASSERT_NO_THROW(region_field->attach(std::move(external_resources), std::move(allocations)));
  ASSERT_TRUE(region_field->is_mapped());
  ASSERT_NO_THROW(region_field->allow_out_of_order_destruction());
}

TEST_F(LogicalRegionFieldDeathTest, AttachFailsOnChild)
{
  auto runtime    = legate::Runtime::get_runtime();
  auto type       = legate::int32();
  auto store      = runtime->create_store(legate::Shape{8}, type);
  auto parent     = store.impl()->get_region_field();
  auto shape      = legate::make_internal_shared<legate::detail::Shape>(1);
  auto field_size = type.size();
  auto child      = legate::detail::LogicalRegionField{
    shape, field_size, parent->region(), parent->field_id() + 1, parent};

  ASSERT_TRUE(child.parent().has_value());

  auto test_buffer    = std::vector<std::int64_t>{8, 0};
  auto buffer_size    = test_buffer.size() * sizeof(test_buffer.front());
  auto realm_resource = std::make_unique<Realm::ExternalMemoryResource>(
    reinterpret_cast<std::uintptr_t>(test_buffer.data()), buffer_size, false /* read_only */
  );
  auto allocation = legate::make_internal_shared<legate::detail::ExternalAllocation>(
    false /* read_only */,
    legate::mapping::StoreTarget::SYSMEM,
    test_buffer.data(),
    buffer_size,
    std::move(realm_resource));

  auto parent_map             = parent->map(legate::mapping::StoreTarget::SYSMEM);
  const auto& physical_region = parent_map.get_physical_region();

  // logical_region_filed with parent is not allowed to attach
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    ASSERT_EXIT(child.attach(physical_region, allocation),
                ::testing::KilledBySignal(SIGABRT),
                ::testing::HasSubstr("!parent().has_value()"));
  }
}

TEST_F(LogicalRegionFieldUnit, ChildDetach)
{
  auto runtime             = legate::Runtime::get_runtime();
  auto type                = legate::uint16();
  auto store               = runtime->create_store(legate::Shape{3}, type);
  auto parent_region_field = store.impl()->get_region_field();
  auto child_shape         = legate::make_internal_shared<legate::detail::Shape>(1);
  auto field_size          = type.size();
  auto child_region_field  = legate::make_internal_shared<legate::detail::LogicalRegionField>(
    child_shape,
    field_size,
    parent_region_field->region(),
    parent_region_field->field_id(),
    parent_region_field);

  ASSERT_THAT([&] { child_region_field->detach(); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Manual detach must be called on the root store")));
}

TEST_F(LogicalRegionFieldUnit, BasicRelease)
{
  auto runtime      = legate::Runtime::get_runtime();
  auto store        = runtime->create_store(legate::Shape{2}, legate::uint16());
  auto region_field = store.impl()->get_region_field();

  region_field->allow_out_of_order_destruction();
  region_field->release_region_field();
  ASSERT_FALSE(region_field->is_mapped());
}

TEST_F(LogicalRegionFieldUnit, MappedRelease)
{
  auto runtime      = legate::Runtime::get_runtime();
  auto store        = runtime->create_store(legate::Shape{1}, legate::int16());
  auto region_field = store.impl()->get_region_field();
  auto map          = region_field->map(legate::mapping::StoreTarget::SYSMEM);

  ASSERT_TRUE(region_field->is_mapped());
  ASSERT_TRUE(map.valid());
  region_field->release_region_field();
  ASSERT_FALSE(region_field->is_mapped());
  ASSERT_TRUE(map.valid());
}

TEST_F(LogicalRegionFieldUnit, ChildRelease)
{
  auto runtime             = legate::Runtime::get_runtime();
  auto type                = legate::int16();
  auto store               = runtime->create_store(legate::Shape{1}, type);
  auto parent_region_field = store.impl()->get_region_field();
  auto child_shape         = legate::make_internal_shared<legate::detail::Shape>(1);
  auto field_size          = type.size();
  auto child_region_field  = legate::make_internal_shared<legate::detail::LogicalRegionField>(
    child_shape,
    field_size,
    parent_region_field->region(),
    parent_region_field->field_id(),
    parent_region_field);

  static_cast<void>(child_region_field->map(legate::mapping::StoreTarget::SYSMEM));
  ASSERT_NO_THROW(child_region_field->release_region_field());
  ASSERT_TRUE(parent_region_field->is_mapped());
  ASSERT_TRUE(child_region_field->is_mapped());
}

TEST_F(LogicalRegionFieldUnit, DoubleChildRelease)
{
  auto runtime             = legate::Runtime::get_runtime();
  auto type                = legate::uint32();
  auto store               = runtime->create_store(legate::Shape{2}, type);
  auto parent_region_field = store.impl()->get_region_field();
  auto child_shape         = legate::make_internal_shared<legate::detail::Shape>(2);
  auto field_size          = type.size();
  auto child_region_field  = legate::make_internal_shared<legate::detail::LogicalRegionField>(
    child_shape,
    field_size,
    parent_region_field->region(),
    parent_region_field->field_id(),
    parent_region_field);

  static_cast<void>(child_region_field->map(legate::mapping::StoreTarget::SYSMEM));
  ASSERT_NO_THROW(child_region_field->release_region_field());
  ASSERT_TRUE(child_region_field->is_mapped());

  // double release
  ASSERT_NO_THROW(child_region_field->release_region_field());
  ASSERT_TRUE(child_region_field->is_mapped());
}

TEST_F(LogicalRegionFieldUnit, InvalidationCallbacks)
{
  auto runtime      = legate::Runtime::get_runtime();
  auto store        = runtime->create_store(legate::Shape{3}, legate::uint32());
  auto region_field = store.impl()->get_region_field();
  std::vector<int> callback_order;

  region_field->add_invalidation_callback([&callback_order]() { callback_order.push_back(1); });
  region_field->add_invalidation_callback([&callback_order]() { callback_order.push_back(2); });
  region_field->add_invalidation_callback([&callback_order]() { callback_order.push_back(3); });
  region_field->perform_invalidation_callbacks();

  ASSERT_THAT(callback_order, ::testing::ElementsAre(1, 2, 3));

  callback_order.clear();

  // add callback again to verify callback list is cleared
  region_field->add_invalidation_callback([&callback_order]() { callback_order.push_back(4); });
  region_field->perform_invalidation_callbacks();
  ASSERT_THAT(callback_order, ::testing::ElementsAre(4));
}

TEST_F(LogicalRegionFieldUnit, ChildInvalidationCallbacks)
{
  auto runtime           = legate::Runtime::get_runtime();
  auto store             = runtime->create_store(legate::Shape{3}, legate::uint32());
  auto root_region_field = store.impl()->get_region_field();
  auto tiling            = legate::detail::create_tiling(
    legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{1},  // tile_shape
    legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{1},  // color_shape
    legate::detail::SmallVector<std::int64_t, LEGATE_MAX_DIM>{}     // offsets (empty)
  );
  auto child_region_field = root_region_field->get_child(tiling.get(), {0}, true);
  std::vector<int> callback_order;

  child_region_field->add_invalidation_callback(
    [&callback_order]() { callback_order.push_back(1); });

  child_region_field->perform_invalidation_callbacks();
  ASSERT_THAT(callback_order, ::testing::ElementsAre(1));
}

TEST_F(LogicalRegionFieldUnit, InvalidationCallbackWithState)
{
  auto runtime      = legate::Runtime::get_runtime();
  auto store        = runtime->create_store(legate::Shape{3}, legate::uint32());
  auto region_field = store.impl()->get_region_field();

  struct CallbackState {
    int value{0};
    bool called{false};
  };

  auto state = std::make_shared<CallbackState>();

  constexpr std::uint32_t value = 42;
  region_field->add_invalidation_callback([state]() {
    state->value  = value;
    state->called = true;
  });
  region_field->perform_invalidation_callbacks();
  ASSERT_TRUE(state->called);
  ASSERT_EQ(state->value, value);
}

}  // namespace logical_region_field_test
