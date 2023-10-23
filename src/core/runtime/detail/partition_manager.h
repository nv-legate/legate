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

#include <unordered_map>

#include "core/data/shape.h"
#include "core/partitioning/restriction.h"

namespace legate {
class Tiling;
class Weighted;
}  // namespace legate

namespace legate::mapping::detail {
struct Machine;
}  // namespace legate::mapping::detail

namespace legate::detail {

class Runtime;

class PartitionManager {
 public:
  PartitionManager(Runtime* runtime);

 public:
  [[nodiscard]] const std::vector<uint32_t>& get_factors(const mapping::detail::Machine& machine);

 public:
  [[nodiscard]] Shape compute_launch_shape(const mapping::detail::Machine& machine,
                                           const Restrictions& restrictions,
                                           const Shape& shape);
  Shape compute_tile_shape(const Shape& extents, const Shape& launch_shape);
  bool use_complete_tiling(const Shape& extents, const Shape& tile_shape) const;

 public:
  Legion::IndexPartition find_index_partition(const Legion::IndexSpace& index_space,
                                              const Tiling& tiling) const;
  Legion::IndexPartition find_index_partition(const Legion::IndexSpace& index_space,
                                              const Weighted& weighted) const;
  Legion::IndexPartition find_image_partition(const Legion::IndexSpace& index_space,
                                              const Legion::LogicalPartition& func_partition,
                                              Legion::FieldID field_id) const;

 public:
  void record_index_partition(const Legion::IndexSpace& index_space,
                              const Tiling& tiling,
                              const Legion::IndexPartition& index_partition);
  void record_index_partition(const Legion::IndexSpace& index_space,
                              const Weighted& weighted,
                              const Legion::IndexPartition& index_partition);
  void record_image_partition(const Legion::IndexSpace& index_space,
                              const Legion::LogicalPartition& func_partition,
                              Legion::FieldID field_id,
                              const Legion::IndexPartition& index_partition);

 public:
  void invalidate_image_partition(const Legion::IndexSpace& index_space,
                                  const Legion::LogicalPartition& func_partition,
                                  Legion::FieldID field_id);

 private:
  int64_t min_shard_volume_;
  std::unordered_map<uint32_t, std::vector<uint32_t>> all_factors_;

 private:
  using TilingCacheKey = std::pair<Legion::IndexSpace, Tiling>;
  std::map<TilingCacheKey, Legion::IndexPartition> tiling_cache_;
  using WeightedCacheKey = std::pair<Legion::IndexSpace, Weighted>;
  std::map<WeightedCacheKey, Legion::IndexPartition> weighted_cache_;
  using ImageCacheKey = std::tuple<Legion::IndexSpace, Legion::LogicalPartition, Legion::FieldID>;
  std::map<ImageCacheKey, Legion::IndexPartition> image_cache_;
};

}  // namespace legate::detail
