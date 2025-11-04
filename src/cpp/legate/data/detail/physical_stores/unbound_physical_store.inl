/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/physical_stores/unbound_physical_store.h>

namespace legate::detail {

inline UnboundRegionField::UnboundRegionField(const Legion::OutputRegion& out,
                                              Legion::FieldID fid,
                                              bool partitioned)
  : partitioned_{partitioned},
    num_elements_{sizeof(std::size_t),
                  find_memory_kind_for_executing_processor(),
                  nullptr /*init_value*/,
                  alignof(std::size_t)},
    out_{out},
    fid_{fid}
{
}

inline UnboundRegionField::UnboundRegionField(UnboundRegionField&& other) noexcept
  : bound_{std::exchange(other.bound_, false)},
    partitioned_{std::exchange(other.partitioned_, false)},
    num_elements_{std::exchange(other.num_elements_, Legion::UntypedDeferredValue{})},
    out_{std::exchange(other.out_, Legion::OutputRegion{})},
    fid_{std::exchange(other.fid_, -1)}
{
}

inline bool UnboundRegionField::is_partitioned() const { return partitioned_; }

inline bool UnboundRegionField::bound() const { return bound_; }

inline void UnboundRegionField::set_bound(bool bound) { bound_ = bound; }

inline const Legion::OutputRegion& UnboundRegionField::get_output_region() const { return out_; }

inline Legion::FieldID UnboundRegionField::get_field_id() const { return fid_; }

// ==========================================================================================

inline UnboundPhysicalStore::UnboundPhysicalStore(
  std::int32_t dim,
  InternalSharedPtr<Type> type,
  UnboundRegionField&& unbound_field,
  InternalSharedPtr<detail::TransformStack> transform)
  : PhysicalStore{dim,
                  std::move(type),
                  GlobalRedopID{-1},
                  std::move(transform),
                  false,
                  false,
                  false},
    unbound_field_{std::move(unbound_field)}
{
}

inline PhysicalStore::Kind UnboundPhysicalStore::kind() const { return Kind::UNBOUND; }

inline bool UnboundPhysicalStore::valid() const { return true; }

inline ReturnValue UnboundPhysicalStore::pack_weight() const
{
  return unbound_field_.pack_weight();
}

inline bool UnboundPhysicalStore::is_partitioned() const { return unbound_field_.is_partitioned(); }

inline std::pair<Legion::OutputRegion, Legion::FieldID> UnboundPhysicalStore::get_output_field()
  const
{
  return {unbound_field_.get_output_region(), unbound_field_.get_field_id()};
}

}  // namespace legate::detail
