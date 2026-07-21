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

Domain PhysicalStore::domain() const
{
  if (!transformed()) {
    return domain_;
  }
  return transform_->transform(domain_);
}

InlineAllocation PhysicalStore::get_inline_allocation() const
{
  // The base pointer and store-space byte strides are computed directly from the storage's
  // allocation and the (possibly transformed) domain; no Realm region instance is involved.
  auto&& [ptr, strides] = get_base_and_strides();

  return {ptr, std::vector<std::size_t>(strides.begin(), strides.end()), target()};
}

std::pair<void*, SmallVector<std::uint64_t, LEGATE_MAX_DIM>> PhysicalStore::get_base_and_strides()
  const
{
  // Base pointer (address of the storage's coordinate origin) and byte strides over the
  // untransformed storage domain (one entry per storage dimension).
  auto base_and_strides       = storage_()->base_and_byte_strides();
  void* const base            = base_and_strides.first;
  auto&& storage_strides      = base_and_strides.second;
  const auto num_storage_dims = storage_strides.size();

  const auto store_lo = domain().lo();

  // The storage coordinate that the store's lower bound maps to, expressed over the storage's
  // dimensions. For a transformed store, invert the transform stack so that coordinate offsets
  // introduced by projections, shifts, etc. are folded into the point. Otherwise the store and
  // storage coordinates coincide (this also covers 0-D stores, which are never transformed).
  const auto storage_point = transformed() ? transform_->invert_point(store_lo) : store_lo;

  // origin = base + sum_j storage_point[j] * storage_strides[j]  (in bytes)
  std::int64_t offset = 0;

  for (std::size_t j = 0; j < num_storage_dims; ++j) {
    offset += storage_point[j] * static_cast<std::int64_t>(storage_strides[j]);
  }
  auto* const origin = static_cast<std::int8_t*>(base) + offset;

  // Re-express the storage's byte strides over the store's (possibly higher/lower-dimensional)
  // domain.
  auto store_strides = transformed() ? transform_->convert_strides(std::move(storage_strides))
                                     : std::move(storage_strides);

  return {origin, std::move(store_strides)};
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
  // Reflect the freshly bound data in this physical store as well.
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
