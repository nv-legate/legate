/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/storage.h>

#include <legate.h>

#include <legate/data/detail/storage_partition.h>
#include <legate/partitioning/detail/partition/no_partition.h>
#include <legate/runtime/detail/runtime.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace test_storage {

using StorageUnit      = DefaultFixture;
using StorageDeathTest = StorageUnit;

TEST_F(StorageUnit, RegionFieldAccessors)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store({4}, legate::int32());
  auto storage       = logical_store.impl()->get_storage();

  // Verify it's a region field backed storage
  ASSERT_EQ(storage->kind(), legate::detail::Storage::Kind::REGION_FIELD);

  auto region_field = storage->get_region_field();
  ASSERT_NE(region_field.get(), nullptr);
}

TEST_F(StorageUnit, FutureBackedStorage)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store(legate::Scalar{std::int32_t{2}, legate::int32()});
  auto storage       = logical_store.impl()->get_storage();

  ASSERT_EQ(storage->kind(), legate::detail::Storage::Kind::FUTURE);

  auto future = storage->get_future();
  ASSERT_TRUE(future.exists());

  storage->set_future(future, /*scalar_offset=*/0);

  auto new_future = storage->get_future();
  ASSERT_TRUE(new_future.exists());
}

TEST_F(StorageUnit, UnboundStorageIsMappedAndUnmap)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto unbound_store = runtime->create_store(legate::int32(), /*dim=*/1);
  auto storage       = unbound_store.impl()->get_storage();

  // Unbound storage should return false for is_mapped()
  ASSERT_TRUE(storage->unbound());
  ASSERT_FALSE(storage->is_mapped());

  ASSERT_NO_THROW(storage->unmap());
}

TEST_F(StorageUnit, GetRootConstWithSharedPtr)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store({4}, legate::int32());
  auto storage       = logical_store.impl()->get_storage();

  // Call the const version of get_root by using const pointer
  const legate::InternalSharedPtr<const legate::detail::Storage> const_storage = storage;
  auto root = const_storage->get_root(const_storage);

  ASSERT_EQ(root.get(), const_storage.get());
}

TEST_F(StorageUnit, GetRootConstNoArg)
{
  auto runtime                                     = legate::Runtime::get_runtime();
  auto logical_store                               = runtime->create_store({4}, legate::int32());
  auto storage                                     = logical_store.impl()->get_storage();
  const legate::detail::Storage* const_storage_ptr = storage.get();
  const legate::detail::Storage* root              = const_storage_ptr->get_root();

  ASSERT_EQ(root, const_storage_ptr);
}

TEST_F(StorageDeathTest, GetFutureOnRegionField)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store({4}, legate::int32());
  auto storage       = logical_store.impl()->get_storage();

  ASSERT_EQ(storage->kind(), legate::detail::Storage::Kind::REGION_FIELD);
  ASSERT_DEATH(static_cast<void>(storage->get_future()), "Cannot get future from RegionField");
}

TEST_F(StorageDeathTest, GetFutureOrFutureMapOnRegionField)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store({4}, legate::int32());
  auto storage       = logical_store.impl()->get_storage();

  ASSERT_EQ(storage->kind(), legate::detail::Storage::Kind::REGION_FIELD);
  ASSERT_DEATH(static_cast<void>(storage->get_future_or_future_map(Legion::Domain{})),
               "Cannot get future from RegionField");
}

TEST_F(StorageDeathTest, GetFutureOrFutureMapOnFuture)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store(legate::Scalar{std::int32_t{6}, legate::int32()});
  auto storage       = logical_store.impl()->get_storage();

  ASSERT_EQ(storage->kind(), legate::detail::Storage::Kind::FUTURE);
  ASSERT_DEATH(static_cast<void>(storage->get_future_or_future_map(Legion::Domain{})),
               "Cannot get future from Future-backed storage");
}

TEST_F(StorageUnit, GetFutureOrFutureMapOnFutureMapInvalidDomain)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store(legate::Scalar{std::int32_t{9}, legate::int32()});
  auto storage       = logical_store.impl()->get_storage();

  ASSERT_EQ(storage->kind(), legate::detail::Storage::Kind::FUTURE);

  // Before conversion, replicated should be false
  ASSERT_FALSE(storage->replicated());

  // Get the future and create a FutureMap from it
  auto future = storage->get_future();

  // Create a 1-point index space and use it to create a FutureMap
  auto&& legate_runtime       = legate::detail::Runtime::get_runtime();
  auto legion_runtime         = legate_runtime.get_legion_runtime();
  auto legion_context         = legate_runtime.get_legion_context();
  const Legion::Domain domain = Legion::Domain{Legion::DomainPoint{0}, Legion::DomainPoint{0}};
  const Legion::IndexSpace is = legion_runtime->create_index_space(legion_context, domain);
  const Legion::FutureMap fm =
    legion_runtime->construct_future_map(legion_context, is, {{0, future}});

  // Convert to FutureMap-backed storage
  storage->set_future_map(fm, /*scalar_offset=*/0);
  ASSERT_EQ(storage->kind(), legate::detail::Storage::Kind::FUTURE_MAP);

  // After conversion from Future to FutureMap, replicated should be true
  ASSERT_TRUE(storage->replicated());
  auto result = storage->get_future_or_future_map(Legion::Domain{});

  ASSERT_TRUE(std::holds_alternative<Legion::Future>(result));
}

TEST_F(StorageDeathTest, SetFutureOnRegionField)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store({4}, legate::int32());
  auto storage       = logical_store.impl()->get_storage();

  ASSERT_EQ(storage->kind(), legate::detail::Storage::Kind::REGION_FIELD);

  // Create a future from another store
  auto future_store = runtime->create_store(legate::Scalar{std::int32_t{3}, legate::int32()});
  auto future       = future_store.impl()->get_storage()->get_future();

  ASSERT_DEATH(storage->set_future(future, /*scalar_offset=*/0),
               "Cannot set a future on a region field-backed");
}

TEST_F(StorageDeathTest, SetFutureMapOnRegionField)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store({4}, legate::int32());
  auto storage       = logical_store.impl()->get_storage();

  ASSERT_EQ(storage->kind(), legate::detail::Storage::Kind::REGION_FIELD);

  // Create a FutureMap
  auto future_store     = runtime->create_store(legate::Scalar{std::int32_t{1}, legate::int32()});
  auto future           = future_store.impl()->get_storage()->get_future();
  auto&& legate_runtime = legate::detail::Runtime::get_runtime();
  auto legion_runtime   = legate_runtime.get_legion_runtime();
  auto legion_context   = legate_runtime.get_legion_context();
  const Legion::Domain domain = Legion::Domain{Legion::DomainPoint{0}, Legion::DomainPoint{0}};
  const Legion::IndexSpace is = legion_runtime->create_index_space(legion_context, domain);
  const Legion::FutureMap fm =
    legion_runtime->construct_future_map(legion_context, is, {{0, future}});

  ASSERT_DEATH(storage->set_future_map(fm, /*scalar_offset=*/0),
               "Cannot set a future map on a region field-backed");
}

TEST_F(StorageDeathTest, SetRegionFieldOnFuture)
{
  auto runtime        = legate::Runtime::get_runtime();
  auto future_store   = runtime->create_store(legate::Scalar{std::int32_t{3}, legate::int32()});
  auto future_storage = future_store.impl()->get_storage();

  ASSERT_EQ(future_storage->kind(), legate::detail::Storage::Kind::FUTURE);

  // Get a LogicalRegionField from a region-field backed storage
  auto region_store   = runtime->create_store({4}, legate::int32());
  auto region_storage = region_store.impl()->get_storage();
  auto region_field   = region_storage->get_region_field();

  // Make a copy of the shared pointer to pass
  auto region_field_copy = region_field;

  ASSERT_DEATH(future_storage->set_region_field(std::move(region_field_copy)),
               "Cannot set a region field on a future-backed");
}

TEST_F(StorageDeathTest, SetRegionFieldOnFutureMap)
{
  auto runtime      = legate::Runtime::get_runtime();
  auto future_store = runtime->create_store(legate::Scalar{std::int32_t{5}, legate::int32()});
  auto storage      = future_store.impl()->get_storage();

  // Convert to FutureMap-backed
  auto future                 = storage->get_future();
  auto&& legate_runtime       = legate::detail::Runtime::get_runtime();
  auto legion_runtime         = legate_runtime.get_legion_runtime();
  auto legion_context         = legate_runtime.get_legion_context();
  const Legion::Domain domain = Legion::Domain{Legion::DomainPoint{0}, Legion::DomainPoint{0}};
  const Legion::IndexSpace is = legion_runtime->create_index_space(legion_context, domain);
  const Legion::FutureMap fm =
    legion_runtime->construct_future_map(legion_context, is, {{0, future}});
  storage->set_future_map(fm, /*scalar_offset=*/0);

  ASSERT_EQ(storage->kind(), legate::detail::Storage::Kind::FUTURE_MAP);

  // Get a LogicalRegionField from a region-field backed storage
  auto region_store      = runtime->create_store({4}, legate::int32());
  auto region_storage    = region_store.impl()->get_storage();
  auto region_field      = region_storage->get_region_field();
  auto region_field_copy = region_field;

  ASSERT_DEATH(storage->set_region_field(std::move(region_field_copy)),
               "Cannot set a region field on a future map-backed");
}

TEST_F(StorageDeathTest, MapOnFuture)
{
  auto runtime      = legate::Runtime::get_runtime();
  auto future_store = runtime->create_store(legate::Scalar{std::int32_t{7}, legate::int32()});
  auto storage      = future_store.impl()->get_storage();

  ASSERT_EQ(storage->kind(), legate::detail::Storage::Kind::FUTURE);
  ASSERT_DEATH(static_cast<void>(storage->map(legate::mapping::StoreTarget::SYSMEM)),
               "Cannot map a future-backed storage");
}

TEST_F(StorageDeathTest, MapOnFutureMap)
{
  auto runtime      = legate::Runtime::get_runtime();
  auto future_store = runtime->create_store(legate::Scalar{std::int32_t{8}, legate::int32()});
  auto storage      = future_store.impl()->get_storage();

  // Convert to FutureMap-backed
  auto future                 = storage->get_future();
  auto&& legate_runtime       = legate::detail::Runtime::get_runtime();
  auto legion_runtime         = legate_runtime.get_legion_runtime();
  auto legion_context         = legate_runtime.get_legion_context();
  const Legion::Domain domain = Legion::Domain{Legion::DomainPoint{0}, Legion::DomainPoint{0}};
  const Legion::IndexSpace is = legion_runtime->create_index_space(legion_context, domain);
  const Legion::FutureMap fm =
    legion_runtime->construct_future_map(legion_context, is, {{0, future}});
  storage->set_future_map(fm, /*scalar_offset=*/0);

  ASSERT_EQ(storage->kind(), legate::detail::Storage::Kind::FUTURE_MAP);
  ASSERT_DEATH(static_cast<void>(storage->map(legate::mapping::StoreTarget::SYSMEM)),
               "Cannot map a future-map-backed storage");
}

TEST_F(StorageUnit, StoragePartitionGetRoot)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store({4}, legate::int32());

  // Partition the store to get a StoragePartition
  auto partitioned_store = logical_store.partition_by_tiling({2});
  auto storage_partition = partitioned_store.impl()->storage_partition();

  // Get the parent storage for comparison
  auto parent_storage = logical_store.impl()->get_storage();

  // Test get_root() const (no args) - storage_partition.cc:30
  const legate::detail::StoragePartition* const_partition = storage_partition.get();
  const legate::detail::Storage* root                     = const_partition->get_root();
  ASSERT_EQ(root, parent_storage.get());

  // Test get_root(InternalSharedPtr<const StoragePartition>&) const - storage_partition.cc:34-38
  const legate::InternalSharedPtr<const legate::detail::StoragePartition> const_sp =
    storage_partition;
  auto root_shared = const_sp->get_root(const_sp);

  ASSERT_EQ(root_shared.get(), parent_storage.get());
}

// Covers: StoragePartition::get_child_storage() error at storage_partition.cc:59
TEST_F(StorageUnit, StoragePartitionGetChildStorageNonTiling)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store({4}, legate::int32());
  auto storage       = logical_store.impl()->get_storage();

  // Create a NoPartition (not a Tiling partition)
  auto no_partition = legate::detail::create_no_partition();

  // Create a StoragePartition with the NoPartition
  auto storage_partition =
    legate::make_internal_shared<legate::detail::StoragePartition>(storage, no_partition, true);

  // Calling get_child_storage should throw because it's not a Tiling partition
  auto test_fn = [&] {
    legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM> color{0};
    static_cast<void>(storage_partition->get_child_storage(storage_partition, std::move(color)));
  };

  ASSERT_THAT(test_fn,
              ::testing::ThrowsMessage<std::runtime_error>(
                ::testing::HasSubstr("Sub-storage is implemented only for tiling")));
}

// Covers: StoragePartition::get_child_data() error at storage_partition.cc:76
TEST_F(StorageUnit, StoragePartitionGetChildDataNonTiling)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store({4}, legate::int32());
  auto storage       = logical_store.impl()->get_storage();

  // Create a NoPartition (not a Tiling partition)
  auto no_partition = legate::detail::create_no_partition();

  // Create a StoragePartition with the NoPartition
  auto storage_partition =
    legate::make_internal_shared<legate::detail::StoragePartition>(storage, no_partition, true);

  // Calling get_child_data should throw because it's not a Tiling partition
  std::array<std::uint64_t, 1> color = {0};
  auto test_fn                       = [&] {
    static_cast<void>(storage_partition->get_child_data(legate::Span<const std::uint64_t>{color}));
  };

  ASSERT_THAT(test_fn,
              ::testing::ThrowsMessage<std::runtime_error>(
                ::testing::HasSubstr("Sub-storage is implemented only for tiling")));
}

}  // namespace test_storage
