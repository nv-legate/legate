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

#include "core/data/physical_store.h"

#include "core/data/detail/physical_store.h"
#include "core/data/physical_array.h"

#include <fmt/format.h>

namespace legate {

void PhysicalStore::bind_untyped_data(Buffer<int8_t, 1>& buffer, const Point<1>& extents) const
{
  check_valid_binding_(true);
  check_buffer_dimension_(1);

  Legion::OutputRegion out;
  Legion::FieldID fid;

  get_output_field_(out, fid);

  out.return_data(DomainPoint{extents}, fid, buffer.get_instance(), false /*check_constraints*/);

  // We will use this value only when the unbound store is 1D
  update_num_elements_(extents[0]);
}

void PhysicalStore::bind_empty_data() const { impl()->bind_empty_data(); }

std::int32_t PhysicalStore::dim() const { return impl()->dim(); }

Type PhysicalStore::type() const { return Type{impl()->type()}; }

Domain PhysicalStore::domain() const { return impl()->domain(); }

InlineAllocation PhysicalStore::get_inline_allocation() const
{
  return impl()->get_inline_allocation();
}

bool PhysicalStore::is_readable() const { return impl()->is_readable(); }

bool PhysicalStore::is_writable() const { return impl()->is_writable(); }

bool PhysicalStore::is_reducible() const { return impl()->is_reducible(); }

bool PhysicalStore::valid() const { return impl() != nullptr && impl()->valid(); }

bool PhysicalStore::transformed() const { return impl()->transformed(); }

bool PhysicalStore::is_future() const { return impl()->is_future(); }

bool PhysicalStore::is_unbound_store() const { return impl()->is_unbound_store(); }

mapping::StoreTarget PhysicalStore::target() const { return impl()->target(); }

PhysicalStore::PhysicalStore(const PhysicalArray& array)
  : impl_{array.nullable()
            ? throw std::invalid_argument{"Nullable array cannot be converted to a store"}
            : array.data().impl()}
{
}

void PhysicalStore::check_accessor_type_(Type::Code code, std::size_t size_of_T) const
{
  // Test exact match for primitive types
  if (code != Type::Code::NIL) {
    throw std::invalid_argument{
      fmt::format("Type mismatch: {} accessor to a {} store. Disable type checking via accessor "
                  "template parameter if this is intended.",
                  primitive_type(code).to_string(),
                  type().to_string())};
  }
  // Test size matches for other types
  if (size_of_T != type().size()) {
    throw std::invalid_argument{
      fmt::format("Type size mismatch: store type {} has size {}, requested type has size {}. "
                  "Disable type checking via accessor template parameter if this is intended.",
                  type().to_string(),
                  type().size(),
                  size_of_T)};
  }
}

void PhysicalStore::check_accessor_dimension_(std::int32_t dim) const
{
  impl()->check_accessor_dimension_(dim);
}

void PhysicalStore::check_buffer_dimension_(std::int32_t dim) const
{
  impl()->check_buffer_dimension_(dim);
}

void PhysicalStore::check_shape_dimension_(std::int32_t dim) const
{
  impl()->check_shape_dimension_(dim);
}

void PhysicalStore::check_valid_binding_(bool bind_buffer) const
{
  impl()->check_valid_binding_(bind_buffer);
}

void PhysicalStore::check_write_access_() const { impl()->check_write_access_(); }

void PhysicalStore::check_reduction_access_() const { impl()->check_reduction_access_(); }

Legion::DomainAffineTransform PhysicalStore::get_inverse_transform_() const
{
  return impl()->get_inverse_transform_();
}

bool PhysicalStore::is_read_only_future_() const { return impl()->is_read_only_future_(); }

std::size_t PhysicalStore::get_field_offset_() const { return impl()->get_field_offset_(); }

const void* PhysicalStore::get_untyped_pointer_from_future_() const
{
  return impl()->get_untyped_pointer_from_future_();
}

void PhysicalStore::get_region_field_(Legion::PhysicalRegion& pr, Legion::FieldID& fid) const
{
  impl()->get_region_field_(pr, fid);
}

GlobalRedopID PhysicalStore::get_redop_id_() const { return impl()->get_redop_id_(); }

const Legion::Future& PhysicalStore::get_future_() const { return impl()->get_future(); }

const Legion::UntypedDeferredValue& PhysicalStore::get_buffer_() const
{
  return impl()->get_buffer();
}

void PhysicalStore::get_output_field_(Legion::OutputRegion& out, Legion::FieldID& fid) const
{
  impl()->get_output_field_(out, fid);
}

void PhysicalStore::update_num_elements_(std::size_t num_elements) const
{
  impl()->update_num_elements_(num_elements);
}

}  // namespace legate
