/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/physical_store.h>

#include <legate/data/detail/logical_store.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/detail/traced_exception.h>

#include <fmt/format.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace legate::detail {

InlineAllocation PhysicalStore::get_inline_allocation() const
{
  const auto field_size = type_->size();
  auto* const ptr = static_cast<std::int8_t*>(base()) + strides().dot(domain().lo()) * field_size;

  std::vector<std::size_t> byte_strides;

  byte_strides.reserve(strides().size());
  for (auto&& stride : strides()) {
    byte_strides.push_back(static_cast<std::size_t>(stride * field_size));
  }
  return {static_cast<void*>(ptr), std::move(byte_strides), target()};
}

InternalSharedPtr<InlineStorage> PhysicalStore::create_output_buffer_storage(
  const Domain& bounds) const
{
  return make_internal_shared<InlineStorage>(bounds, type_->size(), target());
}

void PhysicalStore::bind_data(InternalSharedPtr<InlineStorage> storage, const Domain& domain)
{
  LEGATE_CHECK(unbound_);
  // Update the originating logical store in place so downstream operations observe the produced
  // data. This is the single-rank realization of the unbound-store "return" path.
  originating_store_->bind_buffer(storage, domain);
  // Unbound stores always have an identity transform (every transform op throws on an unbound
  // store), so the store's affine layout is exactly the freshly bound storage's layout.
  auto&& [base, strides] = storage->base_and_element_strides();

  base_           = base;
  strides_        = std::move(strides);
  inline_storage_ = std::move(storage);
  domain_         = domain;
  unbound_        = false;
  readable_       = true;
}

void PhysicalStore::bind_empty_data()
{
  LEGATE_CHECK(unbound_);

  const auto ndim = dim_ == 0 ? std::int32_t{1} : dim_;
  auto lo         = DomainPoint::zeroes(ndim);
  auto hi         = DomainPoint::zeroes(ndim);

  for (std::int32_t i = 0; i < ndim; ++i) {
    lo[i] = 0;
    hi[i] = -1;  // hi < lo => zero-volume (empty) domain
  }

  const Domain empty{std::move(lo), std::move(hi)};

  bind_data(create_output_buffer_storage(empty), empty);
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
