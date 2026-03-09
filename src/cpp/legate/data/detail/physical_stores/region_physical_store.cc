/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/physical_stores/region_physical_store.h>

#include <legate/data/detail/logical_region_field.h>
#include <legate/data/detail/logical_store.h>
#include <legate/data/detail/shape.h>
#include <legate/data/detail/storage.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/detail/tuple.h>

namespace legate::detail {

Domain RegionPhysicalStore::domain() const
{
  auto result = region_field_.domain();
  // The backing Future or RegionField of any LogicalStorage with an empty shape (e.g. (), (1,0,3))
  // will actually have the 1d Domain <0>..<0>. Therefore, if we ever see this Domain on a Future or
  // RegionField, we can't assume it's the "true" one.
  const bool maybe_fake_domain = result.get_dim() == 1 && result.lo() == 0 && result.hi() == 0;
  if (!transform_->identity()) {
    result = transform_->transform(result);
  }
  LEGATE_CHECK(result.get_dim() == dim() || maybe_fake_domain);
  return result;
}

InlineAllocation RegionPhysicalStore::get_inline_allocation() const
{
  if (transformed()) {
    return region_field_.get_inline_allocation(domain(), get_inverse_transform());
  }
  return region_field_.get_inline_allocation();
}

std::optional<Legion::LogicalRegion> RegionPhysicalStore::get_logical_region() const
{
  return get_region_field().first.get_logical_region();
}

std::optional<Legion::FieldID> RegionPhysicalStore::get_field_id() const
{
  return get_region_field().second;
}

InternalSharedPtr<LogicalStore> RegionPhysicalStore::to_logical_store(
  const InternalSharedPtr<PhysicalStore>& self) const
{
  auto [pr, fid]      = get_region_field();
  auto logical_region = pr.get_logical_region();

  auto store_domain = domain();

  auto extents = from_domain(store_domain);

  SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents_copy = extents;
  auto shape_for_region_field = make_internal_shared<Shape>(std::move(extents_copy));
  auto shape_for_storage      = make_internal_shared<Shape>(std::move(extents));

  auto field_size   = type()->size();
  auto region_field = make_internal_shared<LogicalRegionField>(std::move(shape_for_region_field),
                                                               field_size,
                                                               logical_region,
                                                               fid,
                                                               /*parent=*/std::nullopt,
                                                               /*non_owning=*/true);

  region_field->mark_already_mapped();

  auto storage = make_internal_shared<Storage>(
    std::move(shape_for_storage), std::move(region_field), /*provenance=*/"from_physical_store");

  return make_internal_shared<LogicalStore>(std::move(storage), type(), self);
}

}  // namespace legate::detail
