/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/mapping/mapping.h>
#include <legate/utilities/detail/small_vector.h>

#include <cstdint>

namespace legate::detail {

/**
 * @brief Holds the partition placement information for a partition.
 *
 * @details This class is used to hold the partition placement information for a partition. It is
 * used to get the partition color coordinates, the node ID where the partition was executed, and
 * the memory type used for this partition associated with this placement.
 */
class PartitionPlacement {
 public:
  /**
   * @brief Constructor for PartitionPlacement
   *
   * @param partition_color The color of the partition
   * @param node_id The ID of the node where the partition was executed
   * @param memory_type The memory type used for this partition
   */
  PartitionPlacement(SmallVector<std::uint64_t, LEGATE_MAX_DIM> partition_color,
                     std::uint32_t node_id,
                     mapping::StoreTarget memory_type);

  /**
   * @brief Get partition color coordinates (e.g., [0, 1] for a 2D partition) associated with this
   * mapping.
   *
   * @return Partition color coordinates
   */
  [[nodiscard]] Span<const std::uint64_t> partition_color() const;

  /**
   * @brief Get Node ID where this partition color was executed
   *
   * @return Node ID
   */
  [[nodiscard]] std::uint32_t node_id() const;

  /**
   * @brief Get memory type used for this partition associated with this mapping.
   *
   * @return Memory type
   */
  [[nodiscard]] mapping::StoreTarget memory_type() const;

 private:
  /**
   * @brief The color of the partition
   */
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> partition_color_{};
  /**
   * @brief The node ID where the partition was executed
   */
  std::uint32_t node_id_{0};
  /**
   * @brief The memory type used for this partition
   */
  mapping::StoreTarget memory_type_{mapping::StoreTarget::SYSMEM};
};

}  // namespace legate::detail

#include <legate/data/detail/partition_placement.inl>
