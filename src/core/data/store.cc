/* Copyright 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "core/data/store.h"
#include "core/data/detail/store.h"

namespace legate {

void Store::bind_empty_data() { impl_->bind_empty_data(); }

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

void Store::check_accessor_dimension(const int32_t dim) const
{
  impl_->check_accessor_dimension(dim);
}

void Store::check_buffer_dimension(const int32_t dim) const { impl_->check_buffer_dimension(dim); }

void Store::check_shape_dimension(const int32_t dim) const { impl_->check_shape_dimension(dim); }

void Store::check_valid_binding(bool bind_buffer) const { impl_->check_valid_binding(bind_buffer); }

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

void Store::get_output_field(Legion::OutputRegion& out, Legion::FieldID& fid)
{
  impl_->get_output_field(out, fid);
}

void Store::update_num_elements(size_t num_elements) { impl_->update_num_elements(num_elements); }

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
