/* Copyright 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#pragma once

#include <unordered_map>

#include "legion.h"

#include "core/data/shape.h"
#include "core/partitioning/restriction.h"

namespace legate {

class LibraryContext;
class Runtime;
class Tiling;

namespace mapping {

class MachineDesc;

}  // namespace mapping

class PartitionManager {
 public:
  PartitionManager(Runtime* runtime, const LibraryContext* context);

 public:
  const std::vector<uint32_t>& get_factors(const mapping::MachineDesc& machine);

 public:
  Shape compute_launch_shape(const mapping::MachineDesc& machine,
                             const Restrictions& restrictions,
                             const Shape& shape);
  Shape compute_tile_shape(const Shape& extents, const Shape& launch_shape);

 public:
  Legion::IndexPartition find_index_partition(const Legion::IndexSpace& index_space,
                                              const Tiling& tiling) const;
  void record_index_partition(const Legion::IndexSpace& index_space,
                              const Tiling& tiling,
                              const Legion::IndexPartition& index_partition);

 private:
  int64_t min_shard_volume_;
  std::unordered_map<uint32_t, std::vector<uint32_t>> all_factors_;

 private:
  using TilingCacheKey = std::pair<Legion::IndexSpace, Tiling>;
  std::map<TilingCacheKey, Legion::IndexPartition> tiling_cache_;
};

}  // namespace legate
