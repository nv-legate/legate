/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/detail/restriction.h>
#include <legate/utilities/span.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace legate {

class ParallelPolicy;

}  // namespace legate

namespace legate::mapping::detail {

class Machine;

}  // namespace legate::mapping::detail

namespace legate::detail {

class PartitionManager {
 public:
  PartitionManager() = default;

  PartitionManager(const PartitionManager&)            = delete;
  PartitionManager& operator=(const PartitionManager&) = delete;
  PartitionManager(PartitionManager&&)                 = delete;
  PartitionManager& operator=(PartitionManager&&)      = delete;

  [[nodiscard]] Span<const std::uint32_t> get_factors(const mapping::detail::Machine& machine);

  [[nodiscard]] Point compute_launch_shape(const mapping::detail::Machine& machine,
                                           const ParallelPolicy& parallel_policy,
                                           const Restrictions& restrictions,
                                           const Point& shape);
  [[nodiscard]] Point compute_tile_shape(const Point& extents, const Point& launch_shape);

 private:
  std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> all_factors_{};
};

}  // namespace legate::detail
