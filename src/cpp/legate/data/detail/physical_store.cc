/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/physical_store.h>

#include <legate/data/detail/buffer.h>
#include <legate/data/detail/logical_store.h>
#include <legate/data/detail/raw_allocation.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/detail/traced_exception.h>

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace legate::detail {

const InternalSharedPtr<Buffer>& PhysicalStore::get_buffer() const
{
  if (is_unbound_store()) {
    throw detail::TracedException<std::invalid_argument>{
      "The buffer of an unbound store cannot be retrieved"};
  }
  return buffer_;
}

InternalSharedPtr<Buffer> PhysicalStore::create_output_buffer(const Point& extents,
                                                              bool bind_buffer)
{
  auto buffer = Buffer::allocate(type(), Rect::from_extents(extents), target());

  if (bind_buffer) {
    bind_data(buffer, extents);
  }

  return buffer;
}

void PhysicalStore::bind_data(InternalSharedPtr<Buffer> buffer, const Point& extents)
{
  if (!is_unbound_store()) {
    throw detail::TracedException<std::invalid_argument>{"The store already has a bound buffer"};
  }

  auto store_domain = Rect::from_extents(extents);

  originating_store_->bind_buffer(buffer, store_domain);

  // Unbound stores always have an identity transform (every transform op throws on an unbound
  // store), so the store's affine layout is exactly the freshly bound buffer's layout.
  auto&& [base, strides] = buffer->base_and_element_strides();

  base_     = base;
  strides_  = std::move(strides);
  buffer_   = std::move(buffer);
  domain_   = std::move(store_domain);
  unbound_  = false;
  readable_ = true;
}

void PhysicalStore::bind_empty_data()
{
  auto empty_extents = Point::zeroes(dim());
  auto buffer        = Buffer::allocate(type(), Rect::from_extents(empty_extents), target());

  bind_data(std::move(buffer), empty_extents);
}

bool PhysicalStore::is_accessible(mapping::TaskTarget task_target) const
{
  if (task_target == mapping::TaskTarget::CPU) {
    return target() == mapping::StoreTarget::SYSMEM;
  }

  return target() == mapping::StoreTarget::FBMEM || target() == mapping::StoreTarget::ZCMEM;
}

void PhysicalStore::check_shape_dimension(std::int32_t dim) const
{
  if (dim != this->dim() && (this->dim() != 0 || dim != 1)) {
    throw detail::TracedException<std::invalid_argument>{fmt::format(
      "Dimension mismatch: invalid to retrieve a {}-D rect from a {}-D store", dim, this->dim())};
  }
}

void PhysicalStore::check_accessor_type(Type::Code code, std::size_t size_of_T) const
{
  // Test exact match for primitive types
  if (code != Type::Code::NIL) {
    throw detail::TracedException<std::invalid_argument>{
      fmt::format("Type mismatch: {} accessor to a {} store. Disable type checking via accessor "
                  "template parameter if this is intended.",
                  detail::primitive_type(code)->to_string(),
                  type()->to_string())};
  }
  // Test size matches for other types
  if (size_of_T != type()->size()) {
    throw detail::TracedException<std::invalid_argument>{
      fmt::format("Type size mismatch: store type {} has size {}, requested type has size {}. "
                  "Disable type checking via accessor template parameter if this is intended.",
                  type()->to_string(),
                  type()->size(),
                  size_of_T)};
  }
}

void PhysicalStore::check_accessor_dimension(std::int32_t dim) const
{
  if (dim != this->dim() && (this->dim() != 0 || dim != 1)) {
    throw detail::TracedException<std::invalid_argument>{fmt::format(
      "Dimension mismatch: invalid to create a {}-D accessor to a {}-D store", dim, this->dim())};
  }
}

void PhysicalStore::check_write_access() const
{
  if (!is_writable()) {
    throw detail::TracedException<std::invalid_argument>{"Store isn't writable"};
  }
}

void PhysicalStore::check_reduction_access() const
{
  if (!(is_writable() || is_reducible())) {
    throw detail::TracedException<std::invalid_argument>{"Store isn't reducible"};
  }
}

}  // namespace legate::detail
