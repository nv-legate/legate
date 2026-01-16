/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/partition_manager.h>

#include <legate/mapping/detail/machine.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/runtime/detail/library.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/tuning/parallel_policy.h>
#include <legate/utilities/detail/enumerate.h>
#include <legate/utilities/detail/zip.h>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace legate::detail {

namespace {

[[nodiscard]] std::uint64_t min_shard_volume()
{
  const auto& local_machine = Runtime::get_runtime().local_machine();

  // TODO(wonchanl): make these configurable via Scope
  if (local_machine.has_gpus()) {
    // Make sure we can get at least 1M elements on each GPU
    return Runtime::get_runtime().config().min_gpu_chunk();
  }
  if (local_machine.has_omps()) {
    // Make sure we get at least 128K elements on each OpenMP
    return Runtime::get_runtime().config().min_omp_chunk();
  }
  // Make sure we can get at least 8KB elements on each CPU
  return Runtime::get_runtime().config().min_cpu_chunk();
}

}  // namespace

PartitionManager::PartitionManager() : min_shard_volume_{min_shard_volume()}
{
  LEGATE_CHECK(min_shard_volume_ > 0);
}

Span<const std::uint32_t> PartitionManager::get_factors(const mapping::detail::Machine& machine)
{
  const auto curr_num_pieces = machine.count();
  const auto [it, inserted]  = all_factors_.try_emplace(curr_num_pieces);
  auto& factors              = it->second;

  if (inserted) {
    auto remaining_pieces = curr_num_pieces;

    for (auto prime : {11U, 7U, 5U, 3U, 2U}) {
      while (remaining_pieces % prime == 0) {
        factors.push_back(prime);
        remaining_pieces /= prime;
      }
    }
  }
  return factors;
}

namespace {

[[nodiscard]] SmallVector<std::size_t, LEGATE_MAX_DIM> compute_shape_1d(
  std::uint64_t max_pieces, Span<const std::size_t> shape)
{
  SmallVector<std::size_t, LEGATE_MAX_DIM> result;

  result.push_back(std::min(shape.front(), static_cast<std::size_t>(max_pieces)));
  return result;
}

[[nodiscard]] SmallVector<std::size_t, LEGATE_MAX_DIM> compute_shape_2d(
  std::uint64_t max_pieces, Span<const std::size_t> shape)
{
  // Two dimensional so we can use square root to try and generate as square a pieces
  // as possible since most often we will be doing matrix operations with these
  auto nx         = shape.front();
  auto ny         = shape.back();
  const auto swap = nx > ny;

  if (swap) {
    std::swap(nx, ny);
  }

  const auto n = std::sqrt(static_cast<double>(max_pieces * nx) / static_cast<double>(ny));

  // Need to constraint n to be an integer with numpcs % n == 0
  // try rounding n both up and down
  constexpr auto EPSILON = 1e-12;

  auto n1 = std::max<std::uint64_t>(1, static_cast<std::uint64_t>(std::floor(n + EPSILON)));
  while (max_pieces % n1 != 0) {
    --n1;
  }
  auto n2 = std::max<std::uint64_t>(1, static_cast<std::uint64_t>(std::floor(n - EPSILON)));
  while (max_pieces % n2 != 0) {
    ++n2;
  }

  // pick whichever of n1 and n2 gives blocks closest to square
  // i.e. gives the shortest long side
  const auto side1 = std::max(nx / n1, ny / (max_pieces / n1));
  const auto side2 = std::max(nx / n2, ny / (max_pieces / n2));
  const auto px    = static_cast<std::size_t>(side1 <= side2 ? n1 : n2);
  const auto py    = static_cast<std::size_t>(max_pieces / px);
  auto result      = SmallVector<std::size_t, LEGATE_MAX_DIM>{};

  // we need to trim launch space if it is larger than the original shape in one of the
  // dimensions (can happen in testing)
  result.reserve(2);
  if (swap) {
    // If we swapped, then ny holds previous nx, and nx holds previous ny
    LEGATE_ASSERT(ny == shape.front());
    LEGATE_ASSERT(nx == shape.back());
    result.push_back(std::min(py, ny));
    result.push_back(std::min(px, nx));
  } else {
    result.push_back(std::min(px, nx));
    result.push_back(std::min(py, ny));
  }
  return result;
}

[[nodiscard]] SmallVector<std::size_t, LEGATE_MAX_DIM> compute_shape_nd(
  Span<const std::uint32_t> factors,
  std::size_t ndim,
  std::uint64_t max_pieces,
  Span<const std::size_t> shape)
{
  // For higher dimensions we care less about "square"-ness and more about evenly dividing
  // things, compute the prime factors for our number of pieces and then round-robin them
  // onto the shape, with the goal being to keep the last dimension >= 32 for good memory
  // performance on the GPU
  SmallVector<std::size_t, LEGATE_MAX_DIM> result{tags::size_tag, ndim, 1};
  std::size_t factor_prod = 1;

  for (auto&& factor : factors) {
    // Avoid exceeding the maximum number of pieces
    if (factor * factor_prod > static_cast<std::size_t>(max_pieces)) {
      break;
    }

    factor_prod *= factor;

    SmallVector<std::size_t, LEGATE_MAX_DIM> remaining;

    remaining.reserve(shape.size());
    for (std::uint32_t idx = 0; idx < shape.size(); ++idx) {
      remaining.push_back((shape[idx] + result[idx] - 1) / result[idx]);
    }

    const auto big_dim = static_cast<std::uint32_t>(
      std::max_element(remaining.begin(), remaining.end()) - remaining.begin());

    if (big_dim < ndim - 1) {
      // Not the last dimension, so do it
      result[big_dim] *= factor;
      continue;
    }

    // REVIEW: why 32? no idea
    constexpr auto MAGIC_NUMBER = 32;
    // Last dim so see if it still bigger than 32
    if (remaining[big_dim] / factor >= MAGIC_NUMBER) {
      // go ahead and do it
      result[big_dim] *= factor;
      continue;
    }

    // Won't be see if we can do it with one of the other dimensions
    const auto next_big_dim = static_cast<std::uint32_t>(
      std::max_element(remaining.begin(), remaining.end() - 1) - remaining.begin());

    if (remaining[next_big_dim] / factor > 0) {
      result[next_big_dim] *= factor;
      continue;
    }

    // Fine just do it on the last dimension
    result[big_dim] *= factor;
  }
  return result;
}

}  // namespace

SmallVector<std::uint64_t, LEGATE_MAX_DIM> PartitionManager::compute_launch_shape(
  const mapping::detail::Machine& machine,
  const ParallelPolicy& parallel_policy,
  const Restrictions& restrictions,
  Span<const std::uint64_t> shape)
{
  const auto curr_num_pieces = machine.count() * parallel_policy.overdecompose_factor();
  LEGATE_ASSERT(curr_num_pieces > 0);
  // Easy case if we only have one piece: no parallel launch space
  if (1 == curr_num_pieces) {
    return {};
  }

  // If we only have one point then we never do parallel launches
  if (std::all_of(shape.begin(), shape.end(), [](auto extent) { return 1 == extent; })) {
    return {};
  }

  // Prune out any dimensions that are 1
  auto [temp_shape, temp_dims, volume] = restrictions.prune_dimensions(shape);

  // Figure out how many shards we can make with this array
  std::uint64_t max_pieces = (volume + min_shard_volume_ - 1) / min_shard_volume_;
  LEGATE_CHECK(volume == 0 || max_pieces > 0);
  // If we can only make one piece return that now
  if (max_pieces <= 1) {
    return {};
  }

  // TODO(wonchanl): We need a better heuristic
  max_pieces = curr_num_pieces;

  // First compute the N-th root of the number of pieces
  const auto ndim = temp_shape.size();
  LEGATE_CHECK(ndim > 0);
  // Apparently captured structured bindings are only since C++20. Why on earth did the
  // committee not allow this in C++17????
  LEGATE_CPP_VERSION_TODO(20, "Use captured structured bindings instead");
  auto temp_result = [&](SmallVector<std::size_t, LEGATE_MAX_DIM>* shape_2,
                         std::uint64_t volume_2) {
    switch (ndim) {
      case 1: return compute_shape_1d(max_pieces, *shape_2);
      case 2: {
        if (volume_2 < max_pieces) {
          return std::move(*shape_2);
        }
        return compute_shape_2d(max_pieces, *shape_2);
      }
      default: {  // legate-lint: no-switch-default
        return compute_shape_nd(get_factors(machine), ndim, max_pieces, *shape_2);
      }
    }
  }(&temp_shape, volume);

  // Project back onto the original number of dimensions
  LEGATE_CHECK(temp_result.size() == ndim);

  SmallVector<std::uint64_t, LEGATE_MAX_DIM> result{tags::size_tag, shape.size(), 1};

  for (std::uint32_t idx = 0; idx < ndim; ++idx) {
    result[temp_dims[idx]] = temp_result[idx];
  }
  return result;
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> PartitionManager::compute_tile_shape(
  Span<const std::uint64_t> extents, Span<const std::uint64_t> launch_shape)
{
  LEGATE_CHECK(extents.size() == launch_shape.size());
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape;

  tile_shape.reserve(extents.size());
  std::transform(extents.begin(),
                 extents.end(),
                 launch_shape.begin(),
                 std::back_inserter(tile_shape),
                 [](std::uint64_t ext, std::uint64_t shape) { return (ext + shape - 1) / shape; });
  return tile_shape;
}

bool PartitionManager::use_complete_tiling(Span<const std::uint64_t> extents,
                                           Span<const std::uint64_t> tile_shape)
{
  // If it would generate a very large number of elements then
  // we'll apply a heuristic for now and not actually tile it
  // TODO(wonchanl): A better heuristic for this in the future
  constexpr auto MAX_TILES_HEURISTIC  = 1024;
  constexpr auto MAX_PIECES_HEURISTIC = 16;
  const auto num_tiles                = std::transform_reduce(extents.begin(),
                                               extents.end(),
                                               tile_shape.begin(),
                                               std::size_t{1},
                                               std::multiplies<>{},
                                               std::divides<>{});
  const auto num_pieces = static_cast<std::uint64_t>(Runtime::get_runtime().get_machine().count());
  return num_tiles <= MAX_TILES_HEURISTIC || num_tiles <= MAX_PIECES_HEURISTIC * num_pieces;
}

namespace {

template <typename Cache, typename Partition, typename... Rest>
[[nodiscard]] Legion::IndexPartition find_index_partition_impl(
  const Cache& cache,
  const Legion::IndexSpace& index_space,
  const Partition& partition,
  Rest&&... rest)
{
  const auto finder = cache.find({index_space, partition, std::forward<Rest>(rest)...});

  if (finder != cache.end()) {
    return finder->second;
  }
  return Legion::IndexPartition::NO_PART;
}

}  // namespace

Legion::IndexPartition PartitionManager::find_index_partition(const Legion::IndexSpace& index_space,
                                                              const Tiling& tiling) const
{
  return find_index_partition_impl(tiling_cache_, index_space, tiling);
}

Legion::IndexPartition PartitionManager::find_image_partition(
  const Legion::IndexSpace& index_space,
  const Legion::LogicalPartition& func_partition,
  Legion::FieldID field_id,
  ImageComputationHint hint) const
{
  return find_index_partition_impl(image_cache_, index_space, func_partition, field_id, hint);
}

void PartitionManager::record_index_partition(const Legion::IndexSpace& index_space,
                                              const Tiling& tiling,
                                              const Legion::IndexPartition& index_partition)
{
  tiling_cache_[{index_space, tiling}] = index_partition;
}

void PartitionManager::record_image_partition(const Legion::IndexSpace& index_space,
                                              const Legion::LogicalPartition& func_partition,
                                              Legion::FieldID field_id,
                                              ImageComputationHint hint,
                                              const Legion::IndexPartition& index_partition)
{
  image_cache_[{index_space, func_partition, field_id, hint}] = index_partition;
}

void PartitionManager::invalidate_image_partition(const Legion::IndexSpace& index_space,
                                                  const Legion::LogicalPartition& func_partition,
                                                  Legion::FieldID field_id,
                                                  ImageComputationHint hint)
{
  auto finder = image_cache_.find({index_space, func_partition, field_id, hint});

  LEGATE_ASSERT(finder != image_cache_.end());
  image_cache_.erase(finder);
}

}  // namespace legate::detail
