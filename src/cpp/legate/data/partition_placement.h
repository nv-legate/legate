/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/shared_ptr.h>
#include <legate/utilities/span.h>

#include <cstdint>

/**
 * @file
 * @brief Class definition for legate::PartitionPlacement
 */

namespace legate {

namespace mapping {

enum class StoreTarget : std::uint8_t;

}  // namespace mapping

namespace detail {

class PartitionPlacement;

}  // namespace detail

/**
 * @brief Information about how a partition color maps to a specific device
 */
class LEGATE_EXPORT PartitionPlacement {
 public:
  PartitionPlacement() = LEGATE_DEFAULT_WHEN_CYTHON;

  /**
   * @brief Constructor for PartitionPlacement
   *
   * @param impl The implementation of the PartitionPlacement
   */
  explicit PartitionPlacement(InternalSharedPtr<detail::PartitionPlacement> impl);

  /**
   * @brief Get partition color coordinates (e.g., [0, 1] for a 2D partition)
   *
   * @return Partition color coordinates
   */
  [[nodiscard]] Span<const std::uint64_t> partition_color() const;

  /**
   * @brief Get Node ID where this partition color was executed
   *
   * @return Node ID where this partition color was executed
   */
  [[nodiscard]] std::uint32_t node_id() const;

  /**
   * @brief Get memory type used for this partition
   *
   * @return Memory type used for this partition
   */
  [[nodiscard]] mapping::StoreTarget memory_type() const;

 private:
  SharedPtr<detail::PartitionPlacement> impl_;
};

}  // namespace legate
