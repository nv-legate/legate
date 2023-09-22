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

#include "core/data/store.h"

#include "core/data/array.h"
#include "core/data/detail/store.h"

namespace legate {

void Store::bind_empty_data() const { impl_->bind_empty_data(); }

int32_t Store::dim() const { return impl_->dim(); }

Type Store::type() const { return Type(impl_->type()); }

Domain Store::domain() const { return impl_->domain(); }

bool Store::is_readable() const { return impl_->is_readable(); }

bool Store::is_writable() const { return impl_->is_writable(); }

bool Store::is_reducible() const { return impl_->is_reducible(); }

bool Store::valid() const { return impl_ != nullptr && impl_->valid(); }

bool Store::transformed() const { return impl_->transformed(); }

bool Store::is_future() const { return impl_->is_future(); }

bool Store::is_unbound_store() const { return impl_->is_unbound_store(); }

void Store::unmap() { impl_->unmap(); }

Store::Store(const Array& array) : impl_(array.data().impl()) {}

void Store::check_accessor_dimension(const int32_t dim) const
{
  impl_->check_accessor_dimension(dim);
}

void Store::check_buffer_dimension(const int32_t dim) const { impl_->check_buffer_dimension(dim); }

void Store::check_shape_dimension(const int32_t dim) const { impl_->check_shape_dimension(dim); }

void Store::check_valid_binding(bool bind_buffer) const { impl_->check_valid_binding(bind_buffer); }

void Store::check_write_access() const { impl_->check_write_access(); }

void Store::check_reduction_access() const { impl_->check_reduction_access(); }

Legion::DomainAffineTransform Store::get_inverse_transform() const
{
  return impl_->get_inverse_transform();
}

bool Store::is_read_only_future() const { return impl_->is_read_only_future(); }

void Store::get_region_field(Legion::PhysicalRegion& pr, Legion::FieldID& fid) const
{
  impl_->get_region_field(pr, fid);
}

int32_t Store::get_redop_id() const { return impl_->get_redop_id(); }

Legion::Future Store::get_future() const { return impl_->get_future(); }

Legion::UntypedDeferredValue Store::get_buffer() const { return impl_->get_buffer(); }

void Store::get_output_field(Legion::OutputRegion& out, Legion::FieldID& fid) const
{
  impl_->get_output_field(out, fid);
}

void Store::update_num_elements(size_t num_elements) const
{
  impl_->update_num_elements(num_elements);
}

Store::Store() {}

Store::Store(std::shared_ptr<detail::Store> impl) : impl_(std::move(impl)) {}

Store::Store(const Store& other) : impl_(other.impl_) {}

Store& Store::operator=(const Store& other)
{
  impl_ = other.impl_;
  return *this;
}

Store::Store(Store&& other) : impl_(std::move(other.impl_)) {}

Store& Store::operator=(Store&& other)
{
  impl_ = std::move(other.impl_);
  return *this;
}

Store::~Store() {}

}  // namespace legate
