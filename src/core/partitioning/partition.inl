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

#pragma once

#include "core/partitioning/partition.h"

#include <stdexcept>

namespace legate {

inline NoPartition::Kind NoPartition::kind() const { return Kind::NO_PARTITION; }

inline bool NoPartition::is_complete_for(const detail::Storage* /*storage*/) const { return true; }

inline bool NoPartition::satisfies_restrictions(const Restrictions& /*restrictions*/) const
{
  return true;
}

inline bool NoPartition::is_convertible() const { return true; }

inline Legion::LogicalPartition NoPartition::construct(Legion::LogicalRegion /*region*/,
                                                       bool /*complete*/) const
{
  return Legion::LogicalPartition::NO_PART;
}

inline bool NoPartition::has_launch_domain() const { return false; }

inline const tuple<uint64_t>& NoPartition::color_shape() const
{
  assert(false);
  throw std::invalid_argument{"NoPartition doesn't support color_shape"};
}

// ==========================================================================================

inline Tiling::Kind Tiling::kind() const { return Kind::TILING; }

inline bool Tiling::is_convertible() const { return true; }

inline bool Tiling::has_launch_domain() const { return true; }

inline const tuple<uint64_t>& Tiling::tile_shape() const { return tile_shape_; }

inline const tuple<uint64_t>& Tiling::color_shape() const { return color_shape_; }

inline const tuple<int64_t>& Tiling::offsets() const { return offsets_; }

inline bool Tiling::has_color(const tuple<uint64_t>& color) const { return color < color_shape_; }

// ==========================================================================================

inline Weighted::Kind Weighted::kind() const { return Kind::WEIGHTED; }

inline bool Weighted::is_convertible() const { return false; }

inline bool Weighted::is_complete_for(const detail::Storage* /*storage*/) const
{
  // Partition-by-weight partitions are complete by definition
  return true;
}

inline bool Weighted::has_launch_domain() const { return true; }

inline Domain Weighted::launch_domain() const { return color_domain_; }

inline const tuple<uint64_t>& Weighted::color_shape() const { return color_shape_; }

// ==========================================================================================

inline Image::Kind Image::kind() const { return Kind::IMAGE; }

inline bool Image::is_convertible() const { return false; }

// ==========================================================================================

}  // namespace legate
