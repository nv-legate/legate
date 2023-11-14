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

#include "core/data/physical_store.h"

#include "core/data/detail/physical_store.h"
#include "core/data/physical_array.h"

namespace legate {

void PhysicalStore::bind_empty_data() const { impl_->bind_empty_data(); }

int32_t PhysicalStore::dim() const { return impl_->dim(); }

Type PhysicalStore::type() const { return Type(impl_->type()); }

Domain PhysicalStore::domain() const { return impl_->domain(); }

InlineAllocation PhysicalStore::get_inline_allocation() const
{
  return impl_->get_inline_allocation();
}

bool PhysicalStore::is_readable() const { return impl_->is_readable(); }

bool PhysicalStore::is_writable() const { return impl_->is_writable(); }

bool PhysicalStore::is_reducible() const { return impl_->is_reducible(); }

bool PhysicalStore::valid() const { return impl_ != nullptr && impl_->valid(); }

bool PhysicalStore::transformed() const { return impl_->transformed(); }

bool PhysicalStore::is_future() const { return impl_->is_future(); }

bool PhysicalStore::is_unbound_store() const { return impl_->is_unbound_store(); }

PhysicalStore::PhysicalStore() noexcept = default;

PhysicalStore::PhysicalStore(const PhysicalArray& array) : impl_{array.data().impl()} {}

void PhysicalStore::check_accessor_dimension(int32_t dim) const
{
  impl_->check_accessor_dimension(dim);
}

void PhysicalStore::check_buffer_dimension(int32_t dim) const
{
  impl_->check_buffer_dimension(dim);
}

void PhysicalStore::check_shape_dimension(int32_t dim) const { impl_->check_shape_dimension(dim); }

void PhysicalStore::check_valid_binding(bool bind_buffer) const
{
  impl_->check_valid_binding(bind_buffer);
}

void PhysicalStore::check_write_access() const { impl_->check_write_access(); }

void PhysicalStore::check_reduction_access() const { impl_->check_reduction_access(); }

Legion::DomainAffineTransform PhysicalStore::get_inverse_transform() const
{
  return impl_->get_inverse_transform();
}

bool PhysicalStore::is_read_only_future() const { return impl_->is_read_only_future(); }

void PhysicalStore::get_region_field(Legion::PhysicalRegion& pr, Legion::FieldID& fid) const
{
  impl_->get_region_field(pr, fid);
}

int32_t PhysicalStore::get_redop_id() const { return impl_->get_redop_id(); }

Legion::Future PhysicalStore::get_future() const { return impl_->get_future(); }

Legion::UntypedDeferredValue PhysicalStore::get_buffer() const { return impl_->get_buffer(); }

void PhysicalStore::get_output_field(Legion::OutputRegion& out, Legion::FieldID& fid) const
{
  impl_->get_output_field(out, fid);
}

void PhysicalStore::update_num_elements(size_t num_elements) const
{
  impl_->update_num_elements(num_elements);
}

}  // namespace legate
