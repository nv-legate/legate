/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <legate/data/logical_array.h>

#include <legate/data/detail/logical_array.h>
#include <legate/data/detail/user_storage_tracker.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/detail/traced_exception.h>

#include <stdexcept>
#include <vector>

namespace legate {

class LogicalArray::Impl {
 public:
  explicit Impl(InternalSharedPtr<detail::LogicalArray> impl) : impl_{std::move(impl)}
  {
    this->impl()->collect_storage_trackers(trackers_);
  }

  [[nodiscard]] const SharedPtr<detail::LogicalArray>& impl() const noexcept { return impl_; }

 private:
  SharedPtr<detail::LogicalArray> impl_{};
  std::vector<detail::UserStorageTracker> trackers_{};
};

std::uint32_t LogicalArray::dim() const { return impl()->dim(); }

Type LogicalArray::type() const { return Type{impl()->type()}; }

Shape LogicalArray::shape() const { return Shape{impl()->shape()}; }

const tuple<std::uint64_t>& LogicalArray::extents() const { return shape().extents(); }

std::size_t LogicalArray::volume() const { return impl()->volume(); }

bool LogicalArray::unbound() const { return impl()->unbound(); }

bool LogicalArray::nullable() const { return impl()->nullable(); }

bool LogicalArray::nested() const { return impl()->nested(); }

std::uint32_t LogicalArray::num_children() const { return impl()->num_children(); }

LogicalArray LogicalArray::promote(std::int32_t extra_dim, std::size_t dim_size) const
{
  return LogicalArray{impl()->promote(extra_dim, dim_size)};
}

LogicalArray LogicalArray::project(std::int32_t dim, std::int64_t index) const
{
  return LogicalArray{impl()->project(dim, index)};
}

LogicalArray LogicalArray::slice(std::int32_t dim, Slice sl) const
{
  return LogicalArray{impl()->slice(dim, sl)};
}

LogicalArray LogicalArray::transpose(const std::vector<std::int32_t>& axes) const
{
  return LogicalArray{impl()->transpose(axes)};
}

LogicalArray LogicalArray::delinearize(std::int32_t dim,
                                       const std::vector<std::uint64_t>& sizes) const
{
  return LogicalArray{impl()->delinearize(dim, sizes)};
}

LogicalStore LogicalArray::data() const { return LogicalStore{impl()->data()}; }

LogicalStore LogicalArray::null_mask() const { return LogicalStore{impl()->null_mask()}; }

LogicalArray LogicalArray::child(std::uint32_t index) const
{
  return LogicalArray{impl()->child(index)};
}

PhysicalArray LogicalArray::get_physical_array() const
{
  return PhysicalArray{impl()->get_physical_array(false)};
}

ListLogicalArray LogicalArray::as_list_array() const
{
  if (impl()->kind() != detail::ArrayKind::LIST) {
    throw detail::TracedException<std::invalid_argument>{"Array is not a list array"};
  }
  return ListLogicalArray{impl()};
}

StringLogicalArray LogicalArray::as_string_array() const
{
  if (type().code() != Type::Code::STRING) {
    throw detail::TracedException<std::invalid_argument>{"Array is not a string array"};
  }
  return StringLogicalArray{impl()};
}

void LogicalArray::offload_to(mapping::StoreTarget target_mem) const
{
  detail::Runtime::get_runtime()->offload_to(target_mem, impl());
}

LogicalArray::LogicalArray(InternalSharedPtr<detail::LogicalArray> impl)
  : impl_{make_internal_shared<Impl>(std::move(impl))}
{
}

LogicalArray::~LogicalArray() noexcept = default;

LogicalArray::LogicalArray(const LogicalStore& store)
  : LogicalArray{make_internal_shared<detail::BaseLogicalArray>(store.impl())}
{
}

LogicalArray::LogicalArray(const LogicalStore& store, const LogicalStore& null_mask)
  : LogicalArray{make_internal_shared<detail::BaseLogicalArray>(store.impl(), null_mask.impl())}
{
}

const SharedPtr<detail::LogicalArray>& LogicalArray::impl() const { return impl_->impl(); }

// ==========================================================================================

ListLogicalArray::ListLogicalArray(InternalSharedPtr<detail::LogicalArray> impl)
  : LogicalArray{std::move(impl)}
{
}

LogicalArray ListLogicalArray::descriptor() const
{
  return LogicalArray{static_cast<const detail::ListLogicalArray*>(impl().get())->descriptor()};
}

LogicalArray ListLogicalArray::vardata() const
{
  return LogicalArray{static_cast<const detail::ListLogicalArray*>(impl().get())->vardata()};
}

// ==========================================================================================

StringLogicalArray::StringLogicalArray(InternalSharedPtr<detail::LogicalArray> impl)
  : LogicalArray{std::move(impl)}
{
}

LogicalArray StringLogicalArray::offsets() const
{
  return LogicalArray{static_cast<const detail::ListLogicalArray*>(impl().get())->descriptor()};
}

LogicalArray StringLogicalArray::chars() const
{
  return LogicalArray{static_cast<const detail::ListLogicalArray*>(impl().get())->vardata()};
}

}  // namespace legate
