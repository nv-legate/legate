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

#include "core/runtime/partition_manager.h"

#include "core/legate_c.h"
#include "core/mapping/machine.h"
#include "core/partitioning/partition.h"
#include "core/runtime/context.h"
#include "core/runtime/runtime.h"

namespace legate {

PartitionManager::PartitionManager(Runtime* runtime, const LibraryContext* context)
{
  auto mapper_id = context->get_mapper_id();
  min_shard_volume_ =
    runtime->get_tunable<int64_t>(mapper_id, LEGATE_CORE_TUNABLE_MIN_SHARD_VOLUME);

#ifdef DEBUG_LEGATE
  assert(min_shard_volume_ > 0);
#endif
}

const std::vector<uint32_t>& PartitionManager::get_factors(const mapping::MachineDesc& machine)
{
  uint32_t curr_num_pieces = machine.count();

  auto finder = all_factors_.find(curr_num_pieces);
  if (all_factors_.end() == finder) {
    uint32_t remaining_pieces = curr_num_pieces;
    std::vector<uint32_t> factors;
    auto push_factors = [&factors, &remaining_pieces](uint32_t prime) {
      while (remaining_pieces % prime == 0) {
        factors.push_back(prime);
        remaining_pieces /= prime;
      }
    };
    for (uint32_t factor : {11, 7, 5, 3, 2}) push_factors(factor);
    all_factors_.insert({curr_num_pieces, std::move(factors)});
    finder = all_factors_.find(curr_num_pieces);
  }

  return finder->second;
}

Shape PartitionManager::compute_launch_shape(const mapping::MachineDesc& machine,
                                             const Restrictions& restrictions,
                                             const Shape& shape)
{
  uint32_t curr_num_pieces = machine.count();
  // Easy case if we only have one piece: no parallel launch space
  if (1 == curr_num_pieces) return {};

  // If we only have one point then we never do parallel launches
  if (shape.all([](auto extent) { return 1 == extent; })) return {};

  // Prune out any dimensions that are 1
  std::vector<size_t> temp_shape{};
  std::vector<uint32_t> temp_dims{};
  int64_t volume = 1;
  for (uint32_t dim = 0; dim < shape.size(); ++dim) {
    auto extent = shape[dim];
    if (1 == extent || restrictions[dim] == Restriction::FORBID) continue;
    temp_shape.push_back(extent);
    temp_dims.push_back(dim);
    volume *= extent;
  }

  // Figure out how many shards we can make with this array
  int64_t max_pieces = (volume + min_shard_volume_ - 1) / min_shard_volume_;
  assert(max_pieces > 0);
  // If we can only make one piece return that now
  if (1 == max_pieces) return {};

  // Otherwise we need to compute it ourselves
  // TODO: a better heuristic here.
  //       For now if we can make at least two pieces then we will make N pieces.
  max_pieces = curr_num_pieces;

  // First compute the N-th root of the number of pieces
  uint32_t ndim = temp_shape.size();
  assert(ndim > 0);
  std::vector<size_t> temp_result{};

  if (1 == ndim) {
    // Easy one dimensional case
    temp_result.push_back(std::min<size_t>(temp_shape.front(), static_cast<size_t>(max_pieces)));
  } else if (2 == ndim) {
    if (volume < max_pieces) {
      // TBD: Once the max_pieces heuristic is fixed, this should never happen
      temp_result.swap(temp_shape);
    } else {
      // Two dimensional so we can use square root to try and generate as square a pieces
      // as possible since most often we will be doing matrix operations with these
      auto nx   = temp_shape[0];
      auto ny   = temp_shape[1];
      auto swap = nx > ny;
      if (swap) std::swap(nx, ny);
      auto n = std::sqrt(static_cast<double>(max_pieces) * nx / ny);

      // Need to constraint n to be an integer with numpcs % n == 0
      // try rounding n both up and down
      auto n1 = std::max<int64_t>(1, static_cast<int64_t>(std::floor(n + 1e-12)));
      while (max_pieces % n1 != 0) --n1;
      auto n2 = std::max<int64_t>(1, static_cast<int64_t>(std::floor(n - 1e-12)));
      while (max_pieces % n2 != 0) ++n2;

      // pick whichever of n1 and n2 gives blocks closest to square
      // i.e. gives the shortest long side
      auto side1 = std::max(nx / n1, ny / (max_pieces / n1));
      auto side2 = std::max(nx / n2, ny / (max_pieces / n2));
      auto px    = static_cast<size_t>(side1 <= side2 ? n1 : n2);
      auto py    = static_cast<size_t>(max_pieces / px);

      // we need to trim launch space if it is larger than the
      // original shape in one of the dimensions (can happen in
      // testing)
      if (swap) {
        temp_result.push_back(std::min(py, temp_shape[0]));
        temp_result.push_back(std::min(px, temp_shape[1]));
      } else {
        temp_result.push_back(std::min(px, temp_shape[1]));
        temp_result.push_back(std::min(py, temp_shape[0]));
      }
    }
  } else {
    // For higher dimensions we care less about "square"-ness and more about evenly dividing
    // things, compute the prime factors for our number of pieces and then round-robin them
    // onto the shape, with the goal being to keep the last dimension >= 32 for good memory
    // performance on the GPU
    temp_result.resize(ndim);
    std::fill(temp_result.begin(), temp_result.end(), 1);
    size_t factor_prod = 1;
    for (auto factor : get_factors(machine)) {
      // Avoid exceeding the maximum number of pieces
      if (factor * factor_prod > max_pieces) break;

      factor_prod *= factor;

      std::vector<size_t> remaining;
      for (uint32_t idx = 0; idx < temp_shape.size(); ++idx)
        remaining.push_back((temp_shape[idx] + temp_result[idx] - 1) / temp_result[idx]);
      uint32_t big_dim = std::max_element(remaining.begin(), remaining.end()) - remaining.begin();
      if (big_dim < ndim - 1) {
        // Not the last dimension, so do it
        temp_result[big_dim] *= factor;
      } else {
        // Last dim so see if it still bigger than 32
        if (remaining[big_dim] / factor >= 32) {
          // go ahead and do it
          temp_result[big_dim] *= factor;
        } else {
          // Won't be see if we can do it with one of the other dimensions
          uint32_t next_big_dim =
            std::max_element(remaining.begin(), remaining.end() - 1) - remaining.begin();
          if (remaining[next_big_dim] / factor > 0)
            temp_result[next_big_dim] *= factor;
          else
            // Fine just do it on the last dimension
            temp_result[big_dim] *= factor;
        }
      }
    }
  }

  // Project back onto the original number of dimensions
  assert(temp_result.size() == ndim);
  std::vector<size_t> result(shape.size(), 1);
  for (uint32_t idx = 0; idx < ndim; ++idx) result[temp_dims[idx]] = temp_result[idx];

  return Shape(std::move(result));
}

Shape PartitionManager::compute_tile_shape(const Shape& extents, const Shape& launch_shape)
{
  assert(extents.size() == launch_shape.size());
  Shape tile_shape;
  for (uint32_t idx = 0; idx < extents.size(); ++idx) {
    auto x = extents[idx];
    auto y = launch_shape[idx];
    tile_shape.append_inplace((x + y - 1) / y);
  }
  return tile_shape;
}

bool PartitionManager::use_complete_tiling(const Shape& extents, const Shape& tile_shape) const
{
  // If it would generate a very large number of elements then
  // we'll apply a heuristic for now and not actually tile it
  // TODO: A better heuristic for this in the future
  auto num_tiles  = (extents / tile_shape).volume();
  auto num_pieces = Runtime::get_runtime()->get_machine().count();
  return !(num_tiles > 256 && num_tiles > 16 * num_pieces);
}

Legion::IndexPartition PartitionManager::find_index_partition(const Legion::IndexSpace& index_space,
                                                              const Tiling& tiling) const
{
  auto finder = tiling_cache_.find(std::make_pair(index_space, tiling));
  if (finder != tiling_cache_.end())
    return finder->second;
  else
    return Legion::IndexPartition::NO_PART;
}

void PartitionManager::record_index_partition(const Legion::IndexSpace& index_space,
                                              const Tiling& tiling,
                                              const Legion::IndexPartition& index_partition)
{
  tiling_cache_[std::make_pair(index_space, tiling)] = index_partition;
}

}  // namespace legate
