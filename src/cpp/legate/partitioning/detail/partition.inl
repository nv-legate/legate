/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/detail/partition.h>
#include <legate/utilities/detail/traced_exception.h>

#include <stdexcept>

namespace legate::detail {

inline NoPartition::Kind NoPartition::kind() const { return Kind::NO_PARTITION; }

inline bool NoPartition::is_complete_for(const detail::Storage& /*storage*/) const { return true; }

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

inline Span<const std::uint64_t> NoPartition::color_shape() const
{
  throw TracedException<std::invalid_argument>{"NoPartition doesn't support color_shape"};
}

// ==========================================================================================

inline Tiling::Kind Tiling::kind() const { return Kind::TILING; }

inline bool Tiling::is_convertible() const { return true; }

inline bool Tiling::has_launch_domain() const { return true; }

inline Span<const std::uint64_t> Tiling::tile_shape() const { return tile_shape_; }

inline Span<const std::uint64_t> Tiling::color_shape() const { return color_shape_; }

inline Span<const std::int64_t> Tiling::offsets() const { return offsets_; }

inline Span<const std::uint64_t> Tiling::strides() const { return strides_; }

inline bool Tiling::has_color(Span<const std::uint64_t> color) const
{
  return std::equal(
    color.begin(), color.end(), color_shape().begin(), color_shape().end(), std::less<>{});
}

// ==========================================================================================

inline Weighted::Kind Weighted::kind() const { return Kind::WEIGHTED; }

inline bool Weighted::is_convertible() const { return false; }

inline bool Weighted::is_complete_for(const detail::Storage& /*storage*/) const
{
  // Partition-by-weight partitions are complete by definition
  return true;
}

inline bool Weighted::has_launch_domain() const { return true; }

inline Domain Weighted::launch_domain() const { return color_domain_; }

inline Span<const std::uint64_t> Weighted::color_shape() const { return color_shape_; }

// ==========================================================================================

inline Image::Kind Image::kind() const { return Kind::IMAGE; }

inline bool Image::is_convertible() const { return false; }

inline bool Image::is_complete_for(const detail::Storage& /*storage*/) const
{
  // Completeness check for image partitions is expensive, so we give a sound answer
  return false;
}

inline const InternalSharedPtr<detail::LogicalStore>& Image::func() const { return func_; }

}  // namespace legate::detail
