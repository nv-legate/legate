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
    const auto acc = FieldAccessor<DIM>{region_instance, fid, sub_domain};

    return create_(DIM, sub_domain, acc, target);
  }

  template <std::int32_t M, std::int32_t N>
  [[nodiscard]] InlineAllocation operator()(const Realm::RegionInstance& region_instance,
                                            Realm::FieldID fid,
                                            const Domain& sub_domain,
                                            const Legion::AffineTransform<M, N>& transform,
                                            mapping::StoreTarget target) const
  {
    const auto acc = FieldAccessor<N>{region_instance, fid, transform, sub_domain};

    return create_(N, sub_domain, acc, target);
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

void PhysicalStore::check_accessor_store_backing() const {}

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

InternalSharedPtr<LogicalStore> PhysicalStore::to_logical_store(
  const InternalSharedPtr<PhysicalStore>& /*self*/) const
{
  throw TracedException<std::runtime_error>{
    "InlinePhysicalStore does not support to_logical_store()"};
}

}  // namespace legate::detail
