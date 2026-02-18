/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/physical_stores/inline_physical_store.h>

#include <legate/type/type_traits.h>
#include <legate/utilities/dispatch.h>

#include <legion/api/config.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace legate::detail {

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

InlineAllocation InlinePhysicalStore::get_inline_allocation() const
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

Domain InlinePhysicalStore::domain() const
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

}  // namespace legate::detail
