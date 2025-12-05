/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/data/detail/partition_placement.h>
#include <legate/utilities/detail/small_vector.h>

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace legate::detail {

/**
 * @brief Holds the device mappings for a partitioned store
 *
 * @details This class is used to hold the device mappings for a partitioned store. It is used to
 * get the device mapping for a specific partition color, and to generate a detailed table-formatted
 * description of the device mappings.
 */
class PartitionPlacementInfo {
 public:
  /**
   * @brief Constructor for PartitionPlacementInfo
   *
   * @param mappings The device mappings for this partition
   */
  explicit PartitionPlacementInfo(std::vector<PartitionPlacement> placements);

  /**
   * @brief Get all device mappings for this partition
   *
   * @return All device mappings for this partition
   */
  [[nodiscard]] Span<const PartitionPlacement> placements() const;

  /**
   * @brief Get device mapping for a specific partition color
   *
   * @param color Partition color
   * @return Device mapping for the specific partition color
   */
  [[nodiscard]] std::optional<std::reference_wrapper<const PartitionPlacement>>
  get_placement_for_color(Span<const std::uint64_t> color) const;

  /**
   * @brief Generate a detailed table-formatted description
   *
   * @return Detailed table-formatted description
   */
  [[nodiscard]] std::string to_string() const;

 private:
  /**
   * @brief The device mappings for this partition
   */
  std::vector<PartitionPlacement> placements_;
};

}  // namespace legate::detail

#include <legate/data/detail/partition_placement_info.inl>
