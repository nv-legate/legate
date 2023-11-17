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

#include "core/data/logical_array.h"

#include "core/data/detail/logical_array.h"

namespace legate {

int32_t LogicalArray::dim() const { return impl_->dim(); }

Type LogicalArray::type() const { return Type{impl_->type()}; }

const Shape& LogicalArray::extents() const { return impl_->extents(); }

size_t LogicalArray::volume() const { return impl_->volume(); }

bool LogicalArray::unbound() const { return impl_->unbound(); }

bool LogicalArray::nullable() const { return impl_->nullable(); }

bool LogicalArray::nested() const { return impl_->nested(); }

uint32_t LogicalArray::num_children() const { return impl_->num_children(); }

LogicalArray LogicalArray::promote(int32_t extra_dim, size_t dim_size) const
{
  return LogicalArray{impl_->promote(extra_dim, dim_size)};
}

LogicalArray LogicalArray::project(int32_t dim, int64_t index) const
{
  return LogicalArray{impl_->project(dim, index)};
}

LogicalArray LogicalArray::slice(int32_t dim, Slice sl) const
{
  return LogicalArray{impl_->slice(dim, sl)};
}

LogicalArray LogicalArray::transpose(const std::vector<int32_t>& axes) const
{
  return LogicalArray{impl_->transpose(axes)};
}

LogicalArray LogicalArray::delinearize(int32_t dim, const std::vector<uint64_t>& sizes) const
{
  return LogicalArray{impl_->delinearize(dim, sizes)};
}

LogicalStore LogicalArray::data() const { return LogicalStore{impl_->data()}; }

LogicalStore LogicalArray::null_mask() const { return LogicalStore{impl_->null_mask()}; }

LogicalArray LogicalArray::child(uint32_t index) const { return LogicalArray{impl_->child(index)}; }

PhysicalArray LogicalArray::get_physical_array() const
{
  return PhysicalArray{impl_->get_physical_array()};
}

ListLogicalArray LogicalArray::as_list_array() const
{
  if (impl_->kind() != detail::ArrayKind::LIST) {
    throw std::invalid_argument("Array is not a list array");
  }
  return ListLogicalArray{impl_};
}

StringLogicalArray LogicalArray::as_string_array() const
{
  if (type().code() != Type::Code::STRING) {
    throw std::invalid_argument("Array is not a string array");
  }
  return StringLogicalArray{impl_};
}

LogicalArray::LogicalArray(const LogicalStore& store)
  : impl_{std::make_shared<detail::BaseLogicalArray>(store.impl())}
{
}

LogicalArray::LogicalArray(const LogicalStore& store, const LogicalStore& null_mask)
  : impl_{std::make_shared<detail::BaseLogicalArray>(store.impl(), null_mask.impl())}
{
}

LogicalArray ListLogicalArray::descriptor() const
{
  return LogicalArray{static_cast<const detail::ListLogicalArray*>(impl_.get())->descriptor()};
}

LogicalArray ListLogicalArray::vardata() const
{
  return LogicalArray{static_cast<const detail::ListLogicalArray*>(impl_.get())->vardata()};
}

LogicalArray StringLogicalArray::offsets() const
{
  return LogicalArray{static_cast<const detail::ListLogicalArray*>(impl_.get())->descriptor()};
}

LogicalArray StringLogicalArray::chars() const
{
  return LogicalArray{static_cast<const detail::ListLogicalArray*>(impl_.get())->vardata()};
}

}  // namespace legate
