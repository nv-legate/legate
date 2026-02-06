/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/physical_store.h>

#include <legate/data/detail/physical_stores/future_physical_store.h>
#include <legate/data/detail/physical_stores/region_physical_store.h>
#include <legate/data/detail/physical_stores/unbound_physical_store.h>
#include <legate/utilities/detail/traced_exception.h>

#include <fmt/format.h>

#include <cstddef>
#include <cstdint>
#include <cstring>  // std::memcpy
#include <stdexcept>

namespace legate::detail {

bool PhysicalStore::transformed() const { return !transform_->identity(); }

RegionPhysicalStore& PhysicalStore::as_region_store()
{
  return dynamic_cast<RegionPhysicalStore&>(*this);
}

const RegionPhysicalStore& PhysicalStore::as_region_store() const
{
  return dynamic_cast<const RegionPhysicalStore&>(*this);
}

UnboundPhysicalStore& PhysicalStore::as_unbound_store()
{
  return dynamic_cast<UnboundPhysicalStore&>(*this);
}

const UnboundPhysicalStore& PhysicalStore::as_unbound_store() const
{
  return dynamic_cast<const UnboundPhysicalStore&>(*this);
}

FuturePhysicalStore& PhysicalStore::as_future_store()
{
  return dynamic_cast<FuturePhysicalStore&>(*this);
}

const FuturePhysicalStore& PhysicalStore::as_future_store() const
{
  return dynamic_cast<const FuturePhysicalStore&>(*this);
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

void PhysicalStore::check_accessor_store_backing() const
{
  if (dynamic_cast<const UnboundPhysicalStore*>(this)) {
    throw detail::TracedException<std::invalid_argument>{
      "Invalid to create an accessor on an unbound store"};
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

void PhysicalStore::check_scalar_store() const
{
  if (!dynamic_cast<const FuturePhysicalStore*>(this)) {
    throw detail::TracedException<std::invalid_argument>{"Store isn't a scalar store"};
  }
}

void PhysicalStore::check_unbound_store() const
{
  if (!dynamic_cast<const UnboundPhysicalStore*>(this)) {
    throw detail::TracedException<std::invalid_argument>{"Store isn't an unbound store"};
  }
}

Legion::DomainAffineTransform PhysicalStore::get_inverse_transform() const
{
  return transform_->inverse_transform(dim());
}

bool PhysicalStore::on_target(mapping::StoreTarget /*target*/) const { return false; }

}  // namespace legate::detail
