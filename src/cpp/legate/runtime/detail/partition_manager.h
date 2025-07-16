/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/shape.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/partitioning/detail/restriction.h>
#include <legate/utilities/detail/hash.h>
#include <legate/utilities/hash.h>
#include <legate/utilities/span.h>

#include <map>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace legate {

class ParallelPolicy;

}  // namespace legate

namespace legate::mapping::detail {

class Machine;

}  // namespace legate::mapping::detail

namespace legate::detail {

class Runtime;

class PartitionManager {
 public:
  PartitionManager();

  PartitionManager(const PartitionManager&)            = delete;
  PartitionManager& operator=(const PartitionManager&) = delete;
  PartitionManager(PartitionManager&&)                 = delete;
  PartitionManager& operator=(PartitionManager&&)      = delete;

  [[nodiscard]] const std::vector<std::uint32_t>& get_factors(
    const mapping::detail::Machine& machine);

  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> compute_launch_shape(
    const mapping::detail::Machine& machine,
    const ParallelPolicy& parallel_policy,
    const Restrictions& restrictions,
    Span<const std::uint64_t> shape);
  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> compute_tile_shape(
    Span<const std::uint64_t> extents, Span<const std::uint64_t> launch_shape);
  [[nodiscard]] bool use_complete_tiling(Span<const std::uint64_t> extents,
                                         Span<const std::uint64_t> tile_shape);

  [[nodiscard]] Legion::IndexPartition find_index_partition(const Legion::IndexSpace& index_space,
                                                            const Tiling& tiling) const;
  [[nodiscard]] Legion::IndexPartition find_index_partition(const Legion::IndexSpace& index_space,
                                                            const Weighted& weighted) const;
  [[nodiscard]] Legion::IndexPartition find_image_partition(
    const Legion::IndexSpace& index_space,
    const Legion::LogicalPartition& func_partition,
    Legion::FieldID field_id,
    ImageComputationHint hint) const;

  void record_index_partition(const Legion::IndexSpace& index_space,
                              const Tiling& tiling,
                              const Legion::IndexPartition& index_partition);
  void record_index_partition(const Legion::IndexSpace& index_space,
                              const Weighted& weighted,
                              const Legion::IndexPartition& index_partition);
  void record_image_partition(const Legion::IndexSpace& index_space,
                              const Legion::LogicalPartition& func_partition,
                              Legion::FieldID field_id,
                              ImageComputationHint hint,
                              const Legion::IndexPartition& index_partition);

  void invalidate_image_partition(const Legion::IndexSpace& index_space,
                                  const Legion::LogicalPartition& func_partition,
                                  Legion::FieldID field_id,
                                  ImageComputationHint hint);

 private:
  std::int64_t min_shard_volume_{};
  std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> all_factors_{};

  using TilingCacheKey = std::pair<Legion::IndexSpace, Tiling>;
  std::unordered_map<TilingCacheKey, Legion::IndexPartition, hasher<TilingCacheKey>>
    tiling_cache_{};
  using WeightedCacheKey = std::pair<Legion::IndexSpace, Weighted>;
  std::map<WeightedCacheKey, Legion::IndexPartition> weighted_cache_{};
  using ImageCacheKey =
    std::tuple<Legion::IndexSpace, Legion::LogicalPartition, Legion::FieldID, ImageComputationHint>;
  std::map<ImageCacheKey, Legion::IndexPartition> image_cache_{};
};

}  // namespace legate::detail
