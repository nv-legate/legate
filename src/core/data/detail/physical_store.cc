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

#include "core/data/detail/physical_store.h"

#include "core/data/buffer.h"
#include "core/mapping/detail/mapping.h"
#include "core/utilities/dispatch.h"

#include <cstring>  // std::memcpy
#include <stdexcept>
#include <string>

namespace legate::detail {

UnboundRegionField& UnboundRegionField::operator=(UnboundRegionField&& other) noexcept
{
  if (this != &other) {
    bound_        = std::exchange(other.bound_, false);
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
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    if (!bound_) {
      LEGATE_ABORT(
        "Found an uninitialized unbound store. Please make sure you return buffers to all unbound "
        "stores in the task");
    }
  }
  return {num_elements_, sizeof(size_t)};
}

void UnboundRegionField::update_num_elements(std::size_t num_elements)
{
  const AccessorWO<size_t, 1> acc{num_elements_, sizeof(num_elements), false};
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
    throw std::invalid_argument{"Invalid to retrieve the domain of an unbound store"};
  }

  auto result = is_future() ? future_.domain() : region_field_.domain();
  // The backing Future or RegionField of any LogicalStorage with an empty shape (e.g. (), (1,0,3))
  // will actually have the 1d Domain <0>..<0>. Therefore, if we ever see this Domain on a Future or
  // RegionField, we can't assume it's the "true" one.
  const bool maybe_fake_domain = result.get_dim() == 1 && result.lo() == 0 && result.hi() == 0;
  if (!transform_->identity()) {
    result = transform_->transform(result);
  }
  LegateAssert(result.dim == dim() || maybe_fake_domain);
  return result;
}

InlineAllocation PhysicalStore::get_inline_allocation() const
{
  if (is_unbound_store()) {
    throw std::invalid_argument{"Allocation info cannot be retrieved from an unbound store"};
  }

  if (transformed()) {
    if (is_future()) {
      return future_.get_inline_allocation(domain());
    }
    return region_field_.get_inline_allocation(type()->size(), domain(), get_inverse_transform());
  }
  if (is_future()) {
    return future_.get_inline_allocation();
  }
  return region_field_.get_inline_allocation(type()->size());
}

mapping::StoreTarget PhysicalStore::target() const
{
  if (is_unbound_store()) {
    throw std::invalid_argument{"Target of an unbound store cannot be queried"};
  }
  if (is_future()) {
    return future_.target();
  }
  return region_field_.target();
}

void PhysicalStore::bind_empty_data()
{
  check_valid_binding(true);
  unbound_field_.bind_empty_data(dim());
}

void PhysicalStore::check_accessor_dimension(std::int32_t dim) const
{
  if (dim != this->dim() && (this->dim() != 0 || dim != 1)) {
    throw std::invalid_argument{"Dimension mismatch: invalid to create a " + std::to_string(dim) +
                                "-D accessor to a " + std::to_string(this->dim()) + "-D store"};
  }
}

void PhysicalStore::check_buffer_dimension(std::int32_t dim) const
{
  if (dim != this->dim()) {
    throw std::invalid_argument{"Dimension mismatch: invalid to bind a " + std::to_string(dim) +
                                "-D buffer to a " + std::to_string(this->dim()) + "-D store"};
  }
}

void PhysicalStore::check_shape_dimension(std::int32_t dim) const
{
  if (dim != this->dim() && (this->dim() != 0 || dim != 1)) {
    throw std::invalid_argument{"Dimension mismatch: invalid to retrieve a " + std::to_string(dim) +
                                "-D rect from a " + std::to_string(this->dim()) + "-D store"};
  }
}

void PhysicalStore::check_valid_binding(bool bind_buffer) const
{
  if (!is_unbound_store()) {
    throw std::invalid_argument{"Buffer can be bound only to an unbound store"};
  }
  if (bind_buffer && unbound_field_.bound()) {
    throw std::invalid_argument{"A buffer has already been bound to the store"};
  }
}

void PhysicalStore::check_write_access() const
{
  if (!is_writable()) {
    throw std::invalid_argument{"Store isn't writable"};
  }
}

void PhysicalStore::check_reduction_access() const
{
  if (!(is_writable() || is_reducible())) {
    throw std::invalid_argument{"Store isn't reducible"};
  }
}

Legion::DomainAffineTransform PhysicalStore::get_inverse_transform() const
{
  return transform_->inverse_transform(dim());
}

bool PhysicalStore::is_read_only_future() const { return future_.is_read_only(); }

void PhysicalStore::get_region_field(Legion::PhysicalRegion& pr, Legion::FieldID& fid) const
{
  LegateAssert(!(is_future() || is_unbound_store()));
  pr  = region_field_.get_physical_region();
  fid = region_field_.get_field_id();
}

const Legion::Future& PhysicalStore::get_future() const
{
  LegateAssert(is_future());
  return future_.get_future();
}

const Legion::UntypedDeferredValue& PhysicalStore::get_buffer() const
{
  LegateAssert(is_future());
  return future_.get_buffer();
}

void PhysicalStore::get_output_field(Legion::OutputRegion& out, Legion::FieldID& fid)
{
  LegateAssert(is_unbound_store());
  out = unbound_field_.get_output_region();
  fid = unbound_field_.get_field_id();
}

void PhysicalStore::update_num_elements(std::size_t num_elements)
{
  unbound_field_.update_num_elements(num_elements);
  unbound_field_.set_bound(true);
}

}  // namespace legate::detail
