/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/physical_store.h>

#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/dispatch.h>

#include <fmt/format.h>

#include <cstddef>
#include <cstdint>
#include <cstring>  // std::memcpy
#include <stdexcept>

namespace legate::detail {

Domain PhysicalStore::domain() const
{
  // Reference the domain over the root storage as inline storages do not use
  // sub-storages/storage partitions.
  Domain result = domain_;

  // The backing Future or RegionField of any LogicalStorage with an empty shape (e.g. (), (1,0,3))
  // will actually have the 1d Domain <0>..<0>. Therefore, if we ever see this Domain on a Future or
  // RegionField, we can't assume it's the "true" one.
  const bool maybe_fake_domain = result.get_dim() == 1 && result.lo() == 0 && result.hi() == 0;

  if (transformed()) {
    result = transform_->transform(result);
  }
  LEGATE_CHECK(result.get_dim() == dim() || maybe_fake_domain);
  return result;
}

namespace {

class GetInlineAllocationPtr {
  template <std::int32_t DIM>
  using FieldAccessor = Legion::FieldAccessor<LEGION_READ_WRITE,
                                              std::int8_t,
                                              DIM,
                                              coord_t,
                                              Realm::AffineAccessor<std::int8_t, DIM, coord_t>>;

 public:
  template <std::int32_t DIM>
  [[nodiscard]] InlineAllocation operator()(const Realm::RegionInstance& region_instance,
                                            Realm::FieldID fid,
                                            const Domain& sub_domain,
                                            mapping::StoreTarget target) const
  {
    const auto& dm = sanitize_0d_(sub_domain);
    const auto acc = FieldAccessor<DIM>{region_instance, fid, dm};

    return create_(DIM, dm, acc, target);
  }

  template <std::int32_t M, std::int32_t N>
  [[nodiscard]] InlineAllocation operator()(const Realm::RegionInstance& region_instance,
                                            Realm::FieldID fid,
                                            const Domain& sub_domain,
                                            const Legion::AffineTransform<M, N>& transform,
                                            mapping::StoreTarget target) const
  {
    const auto& dm = sanitize_0d_(sub_domain);
    const auto acc = FieldAccessor<N>{region_instance, fid, transform, dm};

    return create_(N, dm, acc, target);
  }

 private:
  template <typename Rect, typename Acc>
  [[nodiscard]] static InlineAllocation create_(std::size_t dim,
                                                const Rect& rect,
                                                const Acc& acc,
                                                mapping::StoreTarget target)
  {
    auto strides = std::vector<std::size_t>(dim, 0);
    void* ptr    = acc.ptr(rect, strides.data());

    return {ptr, std::move(strides), target};
  }

  static const Domain& sanitize_0d_(const Domain& domain)
  {
    static const Domain for_0d = []() noexcept {
      Domain result;

      result.dim = 1;
      return result;
    }();

    return domain.dim == 0 ? for_0d : domain;
  }
};

}  // namespace

InlineAllocation PhysicalStore::get_inline_allocation() const
{
  // Determine dimensionality of the domain. `domain()` could be 0D/empty,
  // so account for this edge case.
  auto&& sub_domain      = domain();
  const auto subdim      = std::max(1, sub_domain.get_dim());
  auto&& [instance, fid] = get_region_instance();

  LEGATE_CHECK(subdim >= 0);

  if (transformed()) {
    const auto transform = get_inverse_transform();

    return double_dispatch(transform.transform.m,
                           transform.transform.n,
                           GetInlineAllocationPtr{},
                           instance,
                           fid,
                           sub_domain,
                           transform,
                           target());
  }

  return dim_dispatch(subdim, GetInlineAllocationPtr{}, instance, fid, sub_domain, target());
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
  SmallVector<std::int64_t, LEGATE_MAX_DIM> storage_point;

  if (transformed()) {
    SmallVector<std::int64_t, LEGATE_MAX_DIM> store_point;

    store_point.reserve(dim());
    for (std::int32_t i = 0; i < dim(); ++i) {
      store_point.push_back(store_lo[i]);
    }
    storage_point = transform_->invert_point(std::move(store_point));
  } else {
    storage_point.reserve(num_storage_dims);
    for (std::size_t j = 0; j < num_storage_dims; ++j) {
      storage_point.push_back(store_lo[static_cast<std::int32_t>(j)]);
    }
  }

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

bool PhysicalStore::is_accessible(mapping::TaskTarget task_target) const
{
  if (task_target == mapping::TaskTarget::CPU) {
    return target() == mapping::StoreTarget::SYSMEM;
  }

  return target() == mapping::StoreTarget::FBMEM || target() == mapping::StoreTarget::ZCMEM;
}

Legion::DomainAffineTransform PhysicalStore::get_inverse_transform() const
{
  return transform_->inverse_transform(dim());
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
