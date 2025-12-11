/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/physical_stores/region_physical_store.h>

#include <utility>

namespace legate::detail {

inline RegionPhysicalStore::RegionPhysicalStore(std::int32_t dim,
                                                InternalSharedPtr<Type> type,
                                                GlobalRedopID redop_id,
                                                RegionField&& region_field,
                                                InternalSharedPtr<detail::TransformStack> transform)
  : PhysicalStore{dim,
                  std::move(type),
                  redop_id,
                  std::move(transform),
                  region_field.is_readable(),
                  region_field.is_writable(),
                  region_field.is_reducible()},
    region_field_{std::move(region_field)}
{
}

inline bool RegionPhysicalStore::valid() const { return region_field_.valid(); }

inline mapping::StoreTarget RegionPhysicalStore::target() const { return region_field_.target(); }

inline bool RegionPhysicalStore::is_partitioned() const { return region_field_.is_partitioned(); }

inline bool RegionPhysicalStore::on_target(mapping::StoreTarget target) const
{
  return region_field_.target() == target;
}

inline std::pair<Legion::PhysicalRegion, Legion::FieldID> RegionPhysicalStore::get_region_field()
  const
{
  return {region_field_.get_physical_region(), region_field_.get_field_id()};
}

}  // namespace legate::detail
