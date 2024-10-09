/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "legate/data/logical_store.h"

#include "legate/data/detail/logical_store.h"
#include "legate/data/detail/user_storage_tracker.h"
#include "legate/data/physical_store.h"

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
  : impl_{make_internal_shared<Impl>(std::move(impl))}
{
}

std::uint32_t LogicalStore::dim() const { return impl()->dim(); }

Type LogicalStore::type() const { return Type{impl()->type()}; }

Shape LogicalStore::shape() const { return Shape{impl()->shape()}; }

std::size_t LogicalStore::volume() const { return impl()->volume(); }

bool LogicalStore::unbound() const { return impl()->unbound(); }

bool LogicalStore::transformed() const { return impl()->transformed(); }

bool LogicalStore::has_scalar_storage() const { return impl()->has_scalar_storage(); }

bool LogicalStore::overlaps(const LogicalStore& other) const
{
  return impl()->overlaps(other.impl());
}

LogicalStore LogicalStore::promote(std::int32_t extra_dim, std::size_t dim_size) const
{
  return LogicalStore{impl()->promote(extra_dim, dim_size)};
}

LogicalStore LogicalStore::project(std::int32_t dim, std::int64_t index) const
{
  return LogicalStore{impl()->project(dim, index)};
}

LogicalStorePartition LogicalStore::partition_by_tiling(std::vector<std::uint64_t> tile_shape) const
{
  return LogicalStorePartition{
    detail::partition_store_by_tiling(impl(), tuple<std::uint64_t>{std::move(tile_shape)})};
}

LogicalStore LogicalStore::slice(std::int32_t dim, Slice sl) const
{
  return LogicalStore{detail::slice_store(impl(), dim, sl)};
}

LogicalStore LogicalStore::transpose(std::vector<std::int32_t>&& axes) const
{
  return LogicalStore{impl()->transpose(std::move(axes))};
}

LogicalStore LogicalStore::delinearize(std::int32_t dim, std::vector<std::uint64_t> sizes) const
{
  return LogicalStore{impl()->delinearize(dim, std::move(sizes))};
}

PhysicalStore LogicalStore::get_physical_store() const
{
  return PhysicalStore{impl()->get_physical_store(/* ignore_future_mutability */ false)};
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

LogicalStore::~LogicalStore() noexcept = default;

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

const tuple<std::uint64_t>& LogicalStorePartition::color_shape() const
{
  return impl()->color_shape();
}

LogicalStore LogicalStorePartition::get_child_store(const tuple<std::uint64_t>& color) const
{
  return LogicalStore{impl()->get_child_store(color)};
}

const SharedPtr<detail::LogicalStorePartition>& LogicalStorePartition::impl() const
{
  return impl_->impl();
}

LogicalStorePartition::~LogicalStorePartition() noexcept = default;

}  // namespace legate
