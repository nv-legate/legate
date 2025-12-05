/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/data/partition_placement.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/shared_ptr.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

/**
 * @file
 * @brief Class definition for legate::PartitionPlacementInfo
 */

namespace legate {

namespace detail {

class PartitionPlacementInfo;

}  // namespace detail

/**
 * @brief Partition placement information for a partitioned store
 */
class LEGATE_EXPORT PartitionPlacementInfo {
 public:
  PartitionPlacementInfo() = LEGATE_DEFAULT_WHEN_CYTHON;

  /**
   * @brief Constructor for PartitionPlacementInfo
   *
   * @param impl The detail implementation of the PartitionPlacementInfo
   */
  explicit PartitionPlacementInfo(InternalSharedPtr<detail::PartitionPlacementInfo> impl);

  /**
   * @brief Get all placement information for this partition
   *
   * @return All placement information for this partition
   */
  [[nodiscard]] std::vector<PartitionPlacement> placements() const;

  /**
   * @brief Get placement information for a specific partition color
   *
   * @param color Partition color
   *
   * @return Placement information for the specific partition color
   */
  [[nodiscard]] std::optional<PartitionPlacement> get_placement_for_color(
    Span<const std::uint64_t> color) const;

  /**
   * @brief Generate a detailed table-formatted description
   *
   * @return Detailed table-formatted description
   */
  [[nodiscard]] std::string to_string() const;

 private:
  SharedPtr<detail::PartitionPlacementInfo> impl_;
};

}  // namespace legate
