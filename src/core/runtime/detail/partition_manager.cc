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

#include "core/runtime/detail/partition_manager.h"

#include "core/legate_c.h"
#include "core/mapping/detail/machine.h"
#include "core/partitioning/partition.h"
#include "core/runtime/detail/library.h"
#include "core/runtime/detail/runtime.h"

#include <algorithm>
#include <cmath>

namespace legate::detail {

PartitionManager::PartitionManager(Runtime* runtime)
{
  auto mapper_id = runtime->core_library()->get_mapper_id();

  min_shard_volume_ =
    runtime->get_tunable<std::int64_t>(mapper_id, LEGATE_CORE_TUNABLE_MIN_SHARD_VOLUME);

  LegateAssert(min_shard_volume_ > 0);
}

const std::vector<std::uint32_t>& PartitionManager::get_factors(
  const mapping::detail::Machine& machine)
{
  auto curr_num_pieces = machine.count();
  auto finder          = all_factors_.find(curr_num_pieces);

  if (all_factors_.end() == finder) {
    std::vector<std::uint32_t> factors;
    auto remaining_pieces = curr_num_pieces;
    auto push_factors     = [&factors, &remaining_pieces](std::uint32_t prime) {
      while (remaining_pieces % prime == 0) {
        factors.push_back(prime);
        remaining_pieces /= prime;
      }
    };
    for (auto factor : {11U, 7U, 5U, 3U, 2U}) {
      push_factors(factor);
    }
    all_factors_.insert({curr_num_pieces, std::move(factors)});
    finder = all_factors_.find(curr_num_pieces);
  }
  return finder->second;
}

tuple<std::uint64_t> PartitionManager::compute_launch_shape(const mapping::detail::Machine& machine,
                                                            const Restrictions& restrictions,
                                                            const tuple<std::uint64_t>& shape)
{
  auto curr_num_pieces = machine.count();
  // Easy case if we only have one piece: no parallel launch space
  if (1 == curr_num_pieces) {
    return {};
  }

  // If we only have one point then we never do parallel launches
  if (shape.all([](auto extent) { return 1 == extent; })) {
    return {};
  }

  // Prune out any dimensions that are 1
  std::vector<std::size_t> temp_shape{};
  std::vector<std::uint32_t> temp_dims{};
  std::int64_t volume = 1;

  temp_dims.reserve(shape.size());
  temp_shape.reserve(shape.size());
  for (std::uint32_t dim = 0; dim < shape.size(); ++dim) {
    auto extent = shape[dim];

    if (1 == extent || restrictions[dim] == Restriction::FORBID) {
      continue;
    }
    temp_shape.push_back(extent);
    temp_dims.push_back(dim);
    volume *= static_cast<std::int64_t>(extent);
  }

  // Figure out how many shards we can make with this array
  std::int64_t max_pieces = (volume + min_shard_volume_ - 1) / min_shard_volume_;
  LegateCheck(volume == 0 || max_pieces > 0);
  // If we can only make one piece return that now
  if (max_pieces <= 1) {
    return {};
  }

  // TODO(wonchanl): We need a better heuristic
  max_pieces = curr_num_pieces;

  // First compute the N-th root of the number of pieces
  const auto ndim = temp_shape.size();
  LegateCheck(ndim > 0);
  std::vector<std::size_t> temp_result{};

  if (1 == ndim) {
    // Easy one dimensional case
    temp_result.push_back(
      std::min<std::size_t>(temp_shape.front(), static_cast<std::size_t>(max_pieces)));
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
      if (swap) {
        std::swap(nx, ny);
      }
      auto n = std::sqrt(static_cast<double>(max_pieces * nx) / static_cast<double>(ny));

      // Need to constraint n to be an integer with numpcs % n == 0
      // try rounding n both up and down
      constexpr auto EPSILON = 1e-12;

      auto n1 = std::max<std::int64_t>(1, static_cast<std::int64_t>(std::floor(n + EPSILON)));
      while (max_pieces % n1 != 0) {
        --n1;
      }
      auto n2 = std::max<std::int64_t>(1, static_cast<std::int64_t>(std::floor(n - EPSILON)));
      while (max_pieces % n2 != 0) {
        ++n2;
      }

      // pick whichever of n1 and n2 gives blocks closest to square
      // i.e. gives the shortest long side
      auto side1 = std::max(nx / n1, ny / (max_pieces / n1));
      auto side2 = std::max(nx / n2, ny / (max_pieces / n2));
      auto px    = static_cast<std::size_t>(side1 <= side2 ? n1 : n2);
      auto py    = static_cast<std::size_t>(max_pieces / px);

      // we need to trim launch space if it is larger than the
      // original shape in one of the dimensions (can happen in
      // testing)
      if (swap) {
        temp_result.push_back(std::min(py, temp_shape[0]));
        temp_result.push_back(std::min(px, temp_shape[1]));
      } else {
        temp_result.push_back(std::min(px, temp_shape[0]));
        temp_result.push_back(std::min(py, temp_shape[1]));
      }
    }
  } else {
    // For higher dimensions we care less about "square"-ness and more about evenly dividing
    // things, compute the prime factors for our number of pieces and then round-robin them
    // onto the shape, with the goal being to keep the last dimension >= 32 for good memory
    // performance on the GPU
    temp_result.resize(ndim);
    std::fill(temp_result.begin(), temp_result.end(), 1);
    std::size_t factor_prod = 1;

    for (auto factor : get_factors(machine)) {
      // Avoid exceeding the maximum number of pieces
      if (factor * factor_prod > static_cast<std::size_t>(max_pieces)) {
        break;
      }

      factor_prod *= factor;

      std::vector<std::size_t> remaining;

      remaining.reserve(temp_shape.size());
      for (std::uint32_t idx = 0; idx < temp_shape.size(); ++idx) {
        remaining.push_back((temp_shape[idx] + temp_result[idx] - 1) / temp_result[idx]);
      }
      const std::uint32_t big_dim =
        std::max_element(remaining.begin(), remaining.end()) - remaining.begin();
      if (big_dim < ndim - 1) {
        // Not the last dimension, so do it
        temp_result[big_dim] *= factor;
      } else {
        // REVIEW: why 32? no idea
        constexpr auto MAGIC_NUMBER = 32;
        // Last dim so see if it still bigger than 32
        if (remaining[big_dim] / factor >= MAGIC_NUMBER) {
          // go ahead and do it
          temp_result[big_dim] *= factor;
        } else {
          // Won't be see if we can do it with one of the other dimensions
          const std::uint32_t next_big_dim =
            std::max_element(remaining.begin(), remaining.end() - 1) - remaining.begin();
          if (remaining[next_big_dim] / factor > 0) {
            temp_result[next_big_dim] *= factor;
          } else {
            // Fine just do it on the last dimension
            temp_result[big_dim] *= factor;
          }
        }
      }
    }
  }

  // Project back onto the original number of dimensions
  LegateCheck(temp_result.size() == ndim);
  std::vector<std::uint64_t> result(shape.size(), 1);
  for (std::uint32_t idx = 0; idx < ndim; ++idx) {
    result[temp_dims[idx]] = temp_result[idx];
  }

  return tuple<std::uint64_t>{std::move(result)};
}

tuple<std::uint64_t> PartitionManager::compute_tile_shape(const tuple<std::uint64_t>& extents,
                                                          const tuple<std::uint64_t>& launch_shape)
{
  LegateCheck(extents.size() == launch_shape.size());
  tuple<std::uint64_t> tile_shape;
  for (std::uint32_t idx = 0; idx < extents.size(); ++idx) {
    auto x = extents[idx];
    auto y = launch_shape[idx];
    tile_shape.append_inplace((x + y - 1) / y);
  }
  return tile_shape;
}

bool PartitionManager::use_complete_tiling(const tuple<std::uint64_t>& extents,
                                           const tuple<std::uint64_t>& tile_shape)
{
  // If it would generate a very large number of elements then
  // we'll apply a heuristic for now and not actually tile it
  // TODO(wonchanl): A better heuristic for this in the future
  constexpr auto MAX_TILES_HEURISTIC  = 256;
  constexpr auto MAX_PIECES_HEURISTIC = 16;
  const auto num_tiles                = (extents / tile_shape).volume();
  const auto num_pieces               = Runtime::get_runtime()->get_machine().count();
  return num_tiles <= MAX_TILES_HEURISTIC || num_tiles <= MAX_PIECES_HEURISTIC * num_pieces;
}

namespace {

template <class Cache, class Partition>
Legion::IndexPartition _find_index_partition(const Cache& cache,
                                             const Legion::IndexSpace& index_space,
                                             const Partition& partition)
{
  auto finder = cache.find({index_space, partition});

  if (finder != cache.end()) {
    return finder->second;
  }
  return Legion::IndexPartition::NO_PART;
}

}  // namespace

Legion::IndexPartition PartitionManager::find_index_partition(const Legion::IndexSpace& index_space,
                                                              const Tiling& tiling) const
{
  return _find_index_partition(tiling_cache_, index_space, tiling);
}

Legion::IndexPartition PartitionManager::find_index_partition(const Legion::IndexSpace& index_space,
                                                              const Weighted& weighted) const
{
  return _find_index_partition(weighted_cache_, index_space, weighted);
}

Legion::IndexPartition PartitionManager::find_image_partition(
  const Legion::IndexSpace& index_space,
  const Legion::LogicalPartition& func_partition,
  Legion::FieldID field_id) const
{
  auto finder = image_cache_.find({index_space, func_partition, field_id});

  if (finder != image_cache_.end()) {
    return finder->second;
  }
  return Legion::IndexPartition::NO_PART;
}

void PartitionManager::record_index_partition(const Legion::IndexSpace& index_space,
                                              const Tiling& tiling,
                                              const Legion::IndexPartition& index_partition)
{
  tiling_cache_[{index_space, tiling}] = index_partition;
}

void PartitionManager::record_index_partition(const Legion::IndexSpace& index_space,
                                              const Weighted& weighted,
                                              const Legion::IndexPartition& index_partition)
{
  weighted_cache_[{index_space, weighted}] = index_partition;
}

void PartitionManager::record_image_partition(const Legion::IndexSpace& index_space,
                                              const Legion::LogicalPartition& func_partition,
                                              Legion::FieldID field_id,
                                              const Legion::IndexPartition& index_partition)
{
  image_cache_[{index_space, func_partition, field_id}] = index_partition;
}

void PartitionManager::invalidate_image_partition(const Legion::IndexSpace& index_space,
                                                  const Legion::LogicalPartition& func_partition,
                                                  Legion::FieldID field_id)
{
  auto finder = image_cache_.find({index_space, func_partition, field_id});

  LegateAssert(finder != image_cache_.end());
  image_cache_.erase(finder);
}

}  // namespace legate::detail
