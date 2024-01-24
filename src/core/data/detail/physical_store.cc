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

#include "core/utilities/dispatch.h"
#include "core/utilities/machine.h"

#include <cstring>  // std::memcpy
#include <stdexcept>
#include <string>

#if LegateDefined(LEGATE_USE_CUDA)
#include "core/cuda/cuda_help.h"
#include "core/cuda/stream_pool.h"
#endif

namespace legate::detail {

RegionField::RegionField(int32_t dim, const Legion::PhysicalRegion& pr, Legion::FieldID fid)
  : dim_{dim},
    pr_{std::make_unique<Legion::PhysicalRegion>(pr)},
    lr_{pr.get_logical_region()},
    fid_{fid}
{
  auto priv  = pr.get_privilege();
  readable_  = static_cast<bool>(priv & LEGION_READ_PRIV);
  writable_  = static_cast<bool>(priv & LEGION_WRITE_PRIV);
  reducible_ = static_cast<bool>(priv & LEGION_REDUCE) || (readable_ && writable_);
}

bool RegionField::valid() const
{
  return pr_ != nullptr && pr_->get_logical_region() != Legion::LogicalRegion::NO_REGION;
}

namespace {

struct get_inline_alloc_fn {
  template <typename Rect, typename Acc>
  InlineAllocation create(const int32_t DIM, const Rect& rect, Acc&& acc)
  {
    std::vector<size_t> strides(DIM, 0);
    auto ptr = const_cast<void*>(static_cast<const void*>(acc.ptr(rect, strides.data())));
    return {ptr, strides};
  }

  template <int32_t DIM>
  InlineAllocation operator()(const Legion::PhysicalRegion& pr,
                              const Legion::FieldID fid,
                              size_t field_size)
  {
    const Rect<DIM> rect{pr};
    return create(
      DIM, rect, AccessorRO<int8_t, DIM>{pr, fid, rect, field_size, false /*check_field_size*/});
  }

  template <int32_t M, int32_t N>
  InlineAllocation operator()(const Legion::PhysicalRegion& pr,
                              const Legion::FieldID fid,
                              const Domain& domain,
                              const Legion::AffineTransform<M, N>& transform,
                              size_t field_size)
  {
    const Rect<N> rect =
      domain.dim > 0 ? Rect<N>{domain} : Rect<N>{Point<N>::ZEROES(), Point<N>::ZEROES()};
    return create(
      N,
      rect,
      AccessorRO<int8_t, N>{pr, fid, transform, rect, field_size, false /*check_field_size*/});
  }
};

}  // namespace

Domain RegionField::domain() const
{
  return Legion::Runtime::get_runtime()->get_index_space_domain(lr_.get_index_space());
}

InlineAllocation RegionField::get_inline_allocation(uint32_t field_size) const
{
  return dim_dispatch(dim_, get_inline_alloc_fn{}, *pr_, fid_, field_size);
}

InlineAllocation RegionField::get_inline_allocation(
  uint32_t field_size, const Domain& domain, const Legion::DomainAffineTransform& transform) const
{
  return double_dispatch(transform.transform.m,
                         transform.transform.n,
                         get_inline_alloc_fn{},
                         *pr_,
                         fid_,
                         domain,
                         transform,
                         field_size);
}

UnboundRegionField& UnboundRegionField::operator=(UnboundRegionField&& other) noexcept
{
  if (this != &other) {
    bound_        = std::exchange(other.bound_, false);
    num_elements_ = std::exchange(other.num_elements_, Legion::UntypedDeferredValue());
    out_          = std::exchange(other.out_, Legion::OutputRegion());
    fid_          = std::exchange(other.fid_, -1);
  }
  return *this;
}

void UnboundRegionField::bind_empty_data(int32_t ndim)
{
  update_num_elements(0);

  DomainPoint extents;
  extents.dim = ndim;

  for (int32_t dim = 0; dim < ndim; ++dim) {
    extents[dim] = 0;
  }
  auto empty_buffer = create_buffer<int8_t>(0);
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

void UnboundRegionField::update_num_elements(size_t num_elements)
{
  const AccessorWO<size_t, 1> acc{num_elements_, sizeof(num_elements), false};
  acc[0] = num_elements;
}

// Silence pass-by-value since Legion::Domain is POD, and the move ctor just does the copy
// anyways. Unfortunately there is no way to check this programatically (e.g. via a
// static_assert).
FutureWrapper::FutureWrapper(bool read_only,
                             uint32_t field_size,
                             const Domain& domain,  // NOLINT(modernize-pass-by-value)
                             Legion::Future future,
                             bool initialize /*= false*/)
  : read_only_{read_only},
    field_size_{field_size},
    domain_{domain},
    future_{std::make_unique<Legion::Future>(std::move(future))}
{
  if (!read_only) {
    if (LegateDefined(LEGATE_USE_DEBUG)) {
      assert(!initialize || future_->get_untyped_size() == field_size);
    }
    auto mem_kind =
      find_memory_kind_for_executing_processor(LegateDefined(LEGATE_NO_FUTURES_ON_FB));
    if (initialize) {
      auto p_init_value = future_->get_buffer(mem_kind);
#if LegateDefined(LEGATE_USE_CUDA)
      if (mem_kind == Memory::Kind::GPU_FB_MEM) {
        // TODO: This should be done by Legion
        buffer_ = Legion::UntypedDeferredValue(field_size, mem_kind);
        const AccessorWO<int8_t, 1> acc{buffer_, field_size, false};
        auto stream = cuda::StreamPool::get_stream_pool().get_stream();
        CHECK_CUDA(
          cudaMemcpyAsync(acc.ptr(0), p_init_value, field_size, cudaMemcpyDeviceToDevice, stream));
      } else
#endif
        buffer_ = Legion::UntypedDeferredValue(field_size, mem_kind, p_init_value);
    } else {
      buffer_ = Legion::UntypedDeferredValue(field_size, mem_kind);
    }
  }
}

namespace {

struct get_inline_alloc_from_future_fn {
  template <int32_t DIM>
  InlineAllocation operator()(const Legion::Future& future, const Domain& domain, size_t field_size)
  {
    const Rect<DIM> rect =
      domain.dim > 0 ? Rect<DIM>{domain} : Rect<DIM>{Point<DIM>::ZEROES(), Point<DIM>::ZEROES()};
    std::vector<size_t> strides(DIM, 0);
    const AccessorRO<int8_t, DIM> acc{
      future, rect, Memory::Kind::NO_MEMKIND, field_size, false /*check_field_size*/};
    auto ptr = const_cast<void*>(static_cast<const void*>(acc.ptr(rect, strides.data())));
    return InlineAllocation{ptr, std::move(strides)};
  }

  template <int32_t DIM>
  InlineAllocation operator()(const Legion::UntypedDeferredValue& value,
                              const Domain& domain,
                              size_t field_size)
  {
    const Rect<DIM> rect =
      domain.dim > 0 ? Rect<DIM>{domain} : Rect<DIM>{Point<DIM>::ZEROES(), Point<DIM>::ZEROES()};
    std::vector<size_t> strides(DIM, 0);
    const AccessorRO<int8_t, DIM> acc{value, rect, field_size, false /*check_field_size*/};
    auto ptr = const_cast<void*>(static_cast<const void*>(acc.ptr(rect, strides.data())));
    return InlineAllocation{ptr, std::move(strides)};
  }
};

}  // namespace

InlineAllocation FutureWrapper::get_inline_allocation(const Domain& domain) const
{
  if (is_read_only()) {
    return dim_dispatch(
      std::max(1, domain.dim), get_inline_alloc_from_future_fn{}, *future_, domain, field_size_);
  }
  return dim_dispatch(
    std::max(1, domain.dim), get_inline_alloc_from_future_fn{}, buffer_, domain, field_size_);
}

InlineAllocation FutureWrapper::get_inline_allocation() const
{
  return get_inline_allocation(domain_);
}

void FutureWrapper::initialize_with_identity(int32_t redop_id)
{
  const auto untyped_acc = AccessorWO<int8_t, 1>{buffer_, field_size_};
  const auto ptr         = untyped_acc.ptr(0);

  auto redop = Legion::Runtime::get_reduction_op(redop_id);
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    assert(redop->sizeof_lhs == field_size_);
  }
  auto identity = redop->identity;
#if LegateDefined(LEGATE_USE_CUDA)
  if (buffer_.get_instance().get_location().kind() == Memory::Kind::GPU_FB_MEM) {
    auto stream = cuda::StreamPool::get_stream_pool().get_stream();
    CHECK_CUDA(cudaMemcpyAsync(ptr, identity, field_size_, cudaMemcpyHostToDevice, stream));
  } else
#endif
    std::memcpy(ptr, identity, field_size_);
}

ReturnValue FutureWrapper::pack() const { return {buffer_, field_size_}; }

Legion::Future FutureWrapper::get_future() const
{
  return future_ != nullptr ? *future_ : Legion::Future{};
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
  if (!transform_->identity()) {
    result = transform_->transform(result);
  }
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    assert(result.dim == dim() || dim() == 0);
  }
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

void PhysicalStore::bind_empty_data()
{
  check_valid_binding(true);
  unbound_field_.bind_empty_data(dim());
}

void PhysicalStore::check_accessor_dimension(int32_t dim) const
{
  if (dim != this->dim() && (this->dim() != 0 || dim != 1)) {
    throw std::invalid_argument{"Dimension mismatch: invalid to create a " + std::to_string(dim) +
                                "-D accessor to a " + std::to_string(this->dim()) + "-D store"};
  }
}

void PhysicalStore::check_buffer_dimension(int32_t dim) const
{
  if (dim != this->dim()) {
    throw std::invalid_argument{"Dimension mismatch: invalid to bind a " + std::to_string(dim) +
                                "-D buffer to a " + std::to_string(this->dim()) + "-D store"};
  }
}

void PhysicalStore::check_shape_dimension(int32_t dim) const
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
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    assert(!(is_future() || is_unbound_store()));
  }
  pr  = region_field_.get_physical_region();
  fid = region_field_.get_field_id();
}

Legion::Future PhysicalStore::get_future() const
{
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    assert(is_future());
  }
  return future_.get_future();
}

Legion::UntypedDeferredValue PhysicalStore::get_buffer() const
{
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    assert(is_future());
  }
  return future_.get_buffer();
}

void PhysicalStore::get_output_field(Legion::OutputRegion& out, Legion::FieldID& fid)
{
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    assert(is_unbound_store());
  }
  out = unbound_field_.get_output_region();
  fid = unbound_field_.get_field_id();
}

void PhysicalStore::update_num_elements(size_t num_elements)
{
  unbound_field_.update_num_elements(num_elements);
  unbound_field_.set_bound(true);
}

}  // namespace legate::detail
