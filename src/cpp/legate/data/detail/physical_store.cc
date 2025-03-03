/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/physical_store.h>

#include <legate/data/buffer.h>
#include <legate/mapping/detail/mapping.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/dispatch.h>

#include <fmt/format.h>

#include <cstring>  // std::memcpy
#include <stdexcept>

namespace legate::detail {

UnboundRegionField& UnboundRegionField::operator=(UnboundRegionField&& other) noexcept
{
  if (this != &other) {
    bound_        = std::exchange(other.bound_, false);
    partitioned_  = std::exchange(other.partitioned_, false);
    num_elements_ = std::exchange(other.num_elements_, Legion::UntypedDeferredValue{});
    out_          = std::exchange(other.out_, Legion::OutputRegion{});
    fid_          = std::exchange(other.fid_, -1);
  }
  return *this;
}

void UnboundRegionField::bind_empty_data(std::int32_t ndim)
{
  update_num_elements(0);

  DomainPoint extents;
  extents.dim = ndim;

  for (std::int32_t dim = 0; dim < ndim; ++dim) {
    extents[dim] = 0;
  }
  auto empty_buffer = create_buffer<std::int8_t>(0);
  out_.return_data(extents, fid_, empty_buffer.get_instance(), false);
  bound_ = true;
}

ReturnValue UnboundRegionField::pack_weight() const
{
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    if (!bound_) {
      LEGATE_ABORT(
        "Found an uninitialized unbound store. Please make sure you return buffers to all unbound "
        "stores in the task");
    }
  }
  return {num_elements_, sizeof(std::size_t), alignof(std::size_t)};
}

void UnboundRegionField::update_num_elements(std::size_t num_elements)
{
  const AccessorWO<std::size_t, 1> acc{num_elements_, sizeof(num_elements), false};
  acc[0] = num_elements;
}

bool PhysicalStore::valid() const
{
  return is_future() || is_unbound_store() || region_field_.valid();
}

bool PhysicalStore::transformed() const { return !transform_->identity(); }

Domain PhysicalStore::domain() const
{
  if (is_unbound_store()) {
    throw TracedException<std::invalid_argument>{
      "Invalid to retrieve the domain of an unbound store"};
  }

  auto result = is_future() ? future_.domain() : region_field_.domain();
  // The backing Future or RegionField of any LogicalStorage with an empty shape (e.g. (), (1,0,3))
  // will actually have the 1d Domain <0>..<0>. Therefore, if we ever see this Domain on a Future or
  // RegionField, we can't assume it's the "true" one.
  const bool maybe_fake_domain = result.get_dim() == 1 && result.lo() == 0 && result.hi() == 0;
  if (!transform_->identity()) {
    result = transform_->transform(result);
  }
  LEGATE_ASSERT(result.dim == dim() || maybe_fake_domain);
  return result;
}

InlineAllocation PhysicalStore::get_inline_allocation() const
{
  if (is_unbound_store()) {
    throw TracedException<std::invalid_argument>{
      "Allocation info cannot be retrieved from an unbound store"};
  }

  if (transformed()) {
    if (is_future()) {
      return future_.get_inline_allocation(domain());
    }
    return region_field_.get_inline_allocation(domain(), get_inverse_transform_());
  }
  if (is_future()) {
    return future_.get_inline_allocation();
  }
  return region_field_.get_inline_allocation();
}

mapping::StoreTarget PhysicalStore::target() const
{
  if (is_unbound_store()) {
    throw TracedException<std::invalid_argument>{"Target of an unbound store cannot be queried"};
  }
  if (is_future()) {
    return future_.target();
  }
  return region_field_.target();
}

void PhysicalStore::bind_empty_data()
{
  check_valid_binding_(true);
  unbound_field_.bind_empty_data(dim());
}

bool PhysicalStore::is_partitioned() const
{
  return (is_unbound_store() && unbound_field_.is_partitioned()) ||
         (!is_future() && region_field_.is_partitioned());
}

void PhysicalStore::check_accessor_dimension_(std::int32_t dim) const
{
  if (dim != this->dim() && (this->dim() != 0 || dim != 1)) {
    throw TracedException<std::invalid_argument>{fmt::format(
      "Dimension mismatch: invalid to create a {}-D accessor to a {}-D store", dim, this->dim())};
  }
}

void PhysicalStore::check_buffer_dimension_(std::int32_t dim) const
{
  if (dim != this->dim()) {
    throw TracedException<std::invalid_argument>{fmt::format(
      "Dimension mismatch: invalid to bind a {}-D buffer to a {}-D store", dim, this->dim())};
  }
}

void PhysicalStore::check_shape_dimension_(std::int32_t dim) const
{
  if (dim != this->dim() && (this->dim() != 0 || dim != 1)) {
    throw TracedException<std::invalid_argument>{fmt::format(
      "Dimension mismatch: invalid to retrieve a {}-D rect from a {}-D store", dim, this->dim())};
  }
}

void PhysicalStore::check_valid_binding_(bool bind_buffer) const
{
  if (!is_unbound_store()) {
    throw TracedException<std::invalid_argument>{"Buffer can be bound only to an unbound store"};
  }
  if (bind_buffer && unbound_field_.bound()) {
    throw TracedException<std::invalid_argument>{"A buffer has already been bound to the store"};
  }
}

void PhysicalStore::check_write_access_() const
{
  if (!is_writable()) {
    throw TracedException<std::invalid_argument>{"Store isn't writable"};
  }
}

void PhysicalStore::check_reduction_access_() const
{
  if (!(is_writable() || is_reducible())) {
    throw TracedException<std::invalid_argument>{"Store isn't reducible"};
  }
}

Legion::DomainAffineTransform PhysicalStore::get_inverse_transform_() const
{
  return transform_->inverse_transform(dim());
}

bool PhysicalStore::is_read_only_future_() const { return future_.is_read_only(); }

std::size_t PhysicalStore::get_field_offset_() const { return future_.field_offset(); }

const void* PhysicalStore::get_untyped_pointer_from_future_() const
{
  return future_.get_untyped_pointer_from_future();
}

std::pair<Legion::PhysicalRegion, Legion::FieldID> PhysicalStore::get_region_field_() const
{
  LEGATE_ASSERT(!(is_future() || is_unbound_store()));
  return {region_field_.get_physical_region(), region_field_.get_field_id()};
}

const Legion::Future& PhysicalStore::get_future() const
{
  LEGATE_ASSERT(is_future());
  return future_.get_future();
}

const Legion::UntypedDeferredValue& PhysicalStore::get_buffer() const
{
  LEGATE_ASSERT(is_future());
  return future_.get_buffer();
}

std::pair<Legion::OutputRegion, Legion::FieldID> PhysicalStore::get_output_field_()
{
  LEGATE_ASSERT(is_unbound_store());
  return {unbound_field_.get_output_region(), unbound_field_.get_field_id()};
}

void PhysicalStore::update_num_elements_(std::size_t num_elements)
{
  unbound_field_.update_num_elements(num_elements);
  unbound_field_.set_bound(true);
}

}  // namespace legate::detail
