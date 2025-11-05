/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/physical_stores/unbound_region_field.h>

#include <utility>

namespace legate::detail {

inline UnboundRegionField::UnboundRegionField(UnboundRegionField&& other) noexcept
  : bound_{std::exchange(other.bound_, false)},
    partitioned_{std::exchange(other.partitioned_, false)},
    num_elements_{std::exchange(other.num_elements_, Legion::UntypedDeferredValue{})},
    out_{std::exchange(other.out_, Legion::OutputRegion{})},
    fid_{std::exchange(other.fid_, -1)}
{
}

inline UnboundRegionField& UnboundRegionField::operator=(UnboundRegionField&& other) noexcept
{
  if (this != &other) {
    bound_        = std::exchange(other.bound_, false);
    partitioned_  = std::exchange(other.partitioned_, false);
    num_elements_ = std::exchange(other.num_elements_, Legion::UntypedDeferredValue{});
    out_          = std::exchange(other.out_, Legion::OutputRegion{});
    fid_          = std::exchange(other.fid_, -1);
  }
  return *this;
}

inline bool UnboundRegionField::is_partitioned() const { return partitioned_; }

inline bool UnboundRegionField::bound() const { return bound_; }

inline void UnboundRegionField::set_bound(bool bound) { bound_ = bound; }

inline const Legion::OutputRegion& UnboundRegionField::get_output_region() const { return out_; }

inline Legion::FieldID UnboundRegionField::get_field_id() const { return fid_; }

}  // namespace legate::detail
