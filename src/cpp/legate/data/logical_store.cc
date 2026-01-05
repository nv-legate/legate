/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/logical_store.h>

#include <legate/data/detail/logical_store.h>
#include <legate/data/detail/logical_store_partition.h>
#include <legate/data/detail/user_storage_tracker.h>
#include <legate/data/logical_array.h>
#include <legate/data/physical_store.h>
#include <legate/mapping/mapping.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/detail/small_vector.h>

namespace legate {

class LogicalStore::Impl {
 public:
  explicit Impl(InternalSharedPtr<detail::LogicalStore> impl)
    : impl_{std::move(impl)}, tracker_{impl_}
  {
  }

  [[nodiscard]] const SharedPtr<detail::LogicalStore>& impl() const noexcept { return impl_; }

 private:
  SharedPtr<detail::LogicalStore> impl_{};
  detail::UserStorageTracker tracker_;
};

// ==========================================================================================

LogicalStore::LogicalStore(InternalSharedPtr<detail::LogicalStore> impl)
  : impl_{legate::make_shared<Impl>(std::move(impl))}
{
}

std::uint32_t LogicalStore::dim() const { return impl()->dim(); }

Type LogicalStore::type() const { return Type{impl()->type()}; }

Shape LogicalStore::shape() const { return Shape{impl()->shape()}; }

tuple<std::uint64_t> LogicalStore::extents() const { return shape().extents(); }

std::size_t LogicalStore::volume() const { return impl()->volume(); }

bool LogicalStore::unbound() const { return impl()->unbound(); }

bool LogicalStore::transformed() const { return impl()->transformed(); }

bool LogicalStore::has_scalar_storage() const { return impl()->has_scalar_storage(); }

bool LogicalStore::overlaps(const LogicalStore& other) const
{
  return impl()->overlaps(other.impl());
}

LogicalStore LogicalStore::reinterpret_as(const Type& type) const
{
  return LogicalStore{impl()->reinterpret_as(type.impl())};
}

LogicalStore LogicalStore::promote(std::int32_t extra_dim, std::size_t dim_size) const
{
  return LogicalStore{impl()->promote(extra_dim, dim_size)};
}

LogicalStore LogicalStore::project(std::int32_t dim, std::int64_t index) const
{
  return LogicalStore{impl()->project(dim, index)};
}

LogicalStore LogicalStore::broadcast(std::int32_t dim, std::size_t dim_size) const
{
  return LogicalStore{impl()->broadcast(dim, dim_size)};
}

std::optional<LogicalStorePartition> LogicalStore::get_partition() const
{
  // We need to flush the scheduling window to make sure the partition is up to date.
  detail::Runtime::get_runtime().flush_scheduling_window();

  auto&& key_partition = impl()->get_current_key_partition();

  if (!key_partition.has_value()) {
    return std::nullopt;
  }

  auto storage_partition =
    create_storage_partition(impl()->get_storage(), *key_partition, /* complete */ std::nullopt);

  return LogicalStorePartition{legate::make_internal_shared<detail::LogicalStorePartition>(
    *key_partition, std::move(storage_partition), impl())};
}

LogicalStorePartition LogicalStore::partition_by_tiling(
  Span<const std::uint64_t> tile_shape, std::optional<Span<const std::uint64_t>> color_shape) const
{
  std::optional<detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>> color_shape_opt =
    color_shape.has_value()
      ? std::make_optional<detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>>(*color_shape)
      : std::nullopt;
  return LogicalStorePartition{detail::partition_store_by_tiling(
    impl(),
    {detail::tags::iterator_tag, tile_shape.begin(), tile_shape.end()},
    std::move(color_shape_opt))};
}

LogicalStore LogicalStore::slice(std::int32_t dim, Slice sl) const
{
  return LogicalStore{detail::slice_store(impl(), dim, sl)};
}

LogicalStore LogicalStore::transpose(std::vector<std::int32_t>&& axes) const
{
  return LogicalStore{
    impl()->transpose(detail::SmallVector<std::int32_t, LEGATE_MAX_DIM>{std::move(axes)})};
}

LogicalStore LogicalStore::delinearize(std::int32_t dim, std::vector<std::uint64_t> sizes) const
{
  return LogicalStore{
    impl()->delinearize(dim, detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{std::move(sizes)})};
}

std::optional<PhysicalStore> LogicalStore::get_cached_physical_store() const
{
  auto mapped = impl()->get_cached_physical_store();

  if (mapped.has_value()) {
    return PhysicalStore{*mapped, *this};
  }
  return std::nullopt;
}

PhysicalStore LogicalStore::get_physical_store(std::optional<mapping::StoreTarget> target) const
{
  const auto sanitized =
    target.value_or(detail::Runtime::get_runtime().local_machine().has_socket_memory()
                      ? mapping::StoreTarget::SOCKETMEM
                      : mapping::StoreTarget::SYSMEM);

  return PhysicalStore{impl()->get_physical_store(sanitized, /* ignore_future_mutability */ false),
                       *this};
}

bool LogicalStore::equal_storage(const LogicalStore& other) const
{
  return impl()->equal_storage(*other.impl());
}

std::string LogicalStore::to_string() const { return impl()->to_string(); }

const SharedPtr<detail::LogicalStore>& LogicalStore::impl() const { return impl_->impl(); }

// This method must be non-const
// NOLINTNEXTLINE(readability-make-member-function-const)
void LogicalStore::detach() { impl()->detach(); }

void LogicalStore::offload_to(mapping::StoreTarget target_mem)
{
  const LogicalArray array{*this};
  array.offload_to(target_mem);
}

LogicalStore::~LogicalStore() noexcept = default;

void LogicalStore::
  allow_out_of_order_destruction()  // NOLINT(readability-make-member-function-const)
{
  impl()->allow_out_of_order_destruction();
}

// ==========================================================================================

class LogicalStorePartition::Impl {
 public:
  explicit Impl(InternalSharedPtr<detail::LogicalStorePartition> impl)
    : impl_{std::move(impl)}, tracker_{impl_->store()}
  {
  }

  [[nodiscard]] const SharedPtr<detail::LogicalStorePartition>& impl() const { return impl_; }

 private:
  SharedPtr<detail::LogicalStorePartition> impl_{};
  // We need to keep a user reference to the storage in each logical store partition. Otherwise, the
  // program may lose all user references to its logical stores but still have some live store
  // partitions, which would lead to incorrect aliasing of storages within the stores of those live
  // store partitions.
  detail::UserStorageTracker tracker_;
};

// ==========================================================================================

LogicalStorePartition::LogicalStorePartition(InternalSharedPtr<detail::LogicalStorePartition> impl)
  : impl_{make_internal_shared<Impl>(std::move(impl))}
{
}

LogicalStore LogicalStorePartition::store() const { return LogicalStore{impl()->store()}; }

tuple<std::uint64_t> LogicalStorePartition::color_shape() const
{
  auto&& color_shape = impl()->color_shape();
  auto vec           = std::vector<std::uint64_t>{color_shape.begin(), color_shape.end()};

  return tuple<std::uint64_t>{std::move(vec)};
}

LogicalStore LogicalStorePartition::get_child_store(Span<const std::uint64_t> color) const
{
  return LogicalStore{
    impl()->get_child_store(detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{color})};
}

PartitionPlacementInfo LogicalStorePartition::get_placement_info() const
{
  auto detail_mapping = impl()->get_placement_info();
  auto detail_ptr     = make_internal_shared<detail::PartitionPlacementInfo>(detail_mapping);

  return PartitionPlacementInfo{std::move(detail_ptr)};
}

const SharedPtr<detail::LogicalStorePartition>& LogicalStorePartition::impl() const
{
  return impl_->impl();
}

LogicalStorePartition::~LogicalStorePartition() noexcept = default;

}  // namespace legate
