/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/physical_stores/region_physical_store.h>

#include <legate/mapping/detail/mapping.h>
#include <legate/mapping/mapping.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/detail/traced_exception.h>

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

}  // namespace legate::detail
