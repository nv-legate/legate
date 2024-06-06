/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "core/data/detail/region_field.h"

#include "core/mapping/detail/mapping.h"
#include "core/utilities/dispatch.h"

namespace legate::detail {

RegionField::RegionField(std::int32_t dim, Legion::PhysicalRegion pr, Legion::FieldID fid)
  : dim_{dim}, pr_{std::move(pr)}, lr_{get_physical_region().get_logical_region()}, fid_{fid}
{
  const auto priv = get_physical_region().get_privilege();
  readable_       = static_cast<bool>(priv & LEGION_READ_PRIV);
  writable_       = static_cast<bool>(priv & LEGION_WRITE_PRIV);
  reducible_      = static_cast<bool>(priv & LEGION_REDUCE) || (is_readable() && is_writable());
}

bool RegionField::valid() const
{
  return pr_.has_value() && pr_->get_logical_region() != Legion::LogicalRegion::NO_REGION;
}

namespace {

class GetInlineAllocFn {
 public:
  template <typename Rect, typename Acc>
  [[nodiscard]] InlineAllocation create(std::int32_t DIM, const Rect& rect, Acc&& acc)
  {
    auto strides = std::vector<std::size_t>(DIM, 0);
    // If the memory pointed to by acc here is ever actually const, then we are in big trouble.
    auto* ptr = const_cast<void*>(static_cast<const void*>(acc.ptr(rect, strides.data())));

    return {ptr, std::move(strides)};
  }

  template <std::int32_t DIM>
  [[nodiscard]] InlineAllocation operator()(const Legion::PhysicalRegion& pr,
                                            Legion::FieldID fid,
                                            std::size_t field_size)
  {
    const Rect<DIM> rect{pr};
    return create(
      DIM,
      rect,
      AccessorRO<std::int8_t, DIM>{pr, fid, rect, field_size, false /*check_field_size*/});
  }

  template <std::int32_t M, std::int32_t N>
  [[nodiscard]] InlineAllocation operator()(const Legion::PhysicalRegion& pr,
                                            Legion::FieldID fid,
                                            const Domain& domain,
                                            const Legion::AffineTransform<M, N>& transform,
                                            std::size_t field_size)
  {
    const Rect<N> rect =
      domain.dim > 0 ? Rect<N>{domain} : Rect<N>{Point<N>::ZEROES(), Point<N>::ZEROES()};
    return create(
      N,
      rect,
      AccessorRO<std::int8_t, N>{pr, fid, transform, rect, field_size, false /*check_field_size*/});
  }
};

}  // namespace

Domain RegionField::domain() const
{
  return Legion::Runtime::get_runtime()->get_index_space_domain(lr_.get_index_space());
}

InlineAllocation RegionField::get_inline_allocation(std::uint32_t field_size) const
{
  return dim_dispatch(dim(), GetInlineAllocFn{}, get_physical_region(), get_field_id(), field_size);
}

InlineAllocation RegionField::get_inline_allocation(
  std::uint32_t field_size,
  const Domain& domain,
  const Legion::DomainAffineTransform& transform) const
{
  return double_dispatch(transform.transform.m,
                         transform.transform.n,
                         GetInlineAllocFn{},
                         get_physical_region(),
                         get_field_id(),
                         domain,
                         transform,
                         field_size);
}

mapping::StoreTarget RegionField::target() const
{
  std::set<Memory> memories;

  get_physical_region().get_memories(memories);
  LEGATE_ASSERT(memories.size() == 1);
  return mapping::detail::to_target(memories.begin()->kind());
}

}  // namespace legate::detail
