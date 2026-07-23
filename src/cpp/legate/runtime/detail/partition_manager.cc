/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/partition_manager.h>

#include <legate/mapping/detail/machine.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/tuning/parallel_policy.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iterator>
#include <numeric>
#include <utility>

namespace legate::detail {

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

[[nodiscard]] Point compute_shape_1d(std::uint64_t max_pieces, const Point& shape)
{
  return Point::full(1, std::min(shape.front(), static_cast<std::int64_t>(max_pieces)));
}

[[nodiscard]] Point compute_shape_2d(std::uint64_t max_pieces, const Point& shape)
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

  auto n1 = std::max<std::int64_t>(1, std::floor(n + EPSILON));
  while (max_pieces % n1 != 0) {
    --n1;
  }
  auto n2 = std::max<std::int64_t>(1, std::floor(n - EPSILON));
  while (max_pieces % n2 != 0) {
    ++n2;
  }

  // pick whichever of n1 and n2 gives blocks closest to square
  // i.e. gives the shortest long side
  const auto mp    = static_cast<std::int64_t>(max_pieces);
  const auto side1 = std::max(nx / n1, ny / (mp / n1));
  const auto side2 = std::max(nx / n2, ny / (mp / n2));
  const auto px    = side1 <= side2 ? n1 : n2;
  const auto py    = mp / px;

  // we need to trim launch space if it is larger than the original shape in one of the
  // dimensions (can happen in testing)
  if (swap) {
    // If we swapped, then ny holds previous nx, and nx holds previous ny
    LEGATE_ASSERT(ny == shape.front());
    LEGATE_ASSERT(nx == shape.back());
    return Point({std::min(py, ny), std::min(px, nx)});
  }

  return Point{std::min(px, nx), std::min(py, ny)};
}

[[nodiscard]] Point compute_shape_nd(Span<const std::uint32_t> factors,
                                     std::size_t ndim,
                                     std::uint64_t max_pieces,
                                     const Point& shape)
{
  // For higher dimensions we care less about "square"-ness and more about evenly dividing
  // things, compute the prime factors for our number of pieces and then round-robin them
  // onto the shape, with the goal being to keep the last dimension >= 32 for good memory
  // performance on the GPU
  auto result             = Point::full(static_cast<std::int32_t>(ndim), 1);
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

Point PartitionManager::compute_launch_shape(const mapping::detail::Machine& machine,
                                             const ParallelPolicy& parallel_policy,
                                             const Restrictions& restrictions,
                                             const Point& shape)
{
  const auto curr_num_pieces = machine.count() * parallel_policy.overdecompose_factor();
  LEGATE_ASSERT(curr_num_pieces > 0);

  // Easy case if we only have one piece: no parallel launch space
  if (1 == curr_num_pieces) {
    return {};
  }

  // Calculate the total volume of the shape and don't partition it if it's smaller than the
  // threshold
  const auto total_volume     = shape.volume();
  const auto min_shard_volume = parallel_policy.partitioning_threshold(machine.preferred_target());

  if (total_volume < static_cast<std::int64_t>(min_shard_volume)) {
    return {};
  }

  // Always partition stores bigger than the threshold across all processors in the scope
  // TODO(wonchanl): We may need better heuristics here or at least document the behavior
  // accurately (e.g., we name the environment variables "minimum chunk sizes", but the current
  // heuristic don't guarantee a per-processor minimum allocation size.)
  const auto max_pieces = curr_num_pieces;

  // Prune out any dimensions that are 1 or must not be partitioned
  auto [temp_shape, temp_dims, volume] = restrictions.prune_dimensions(shape);
  const auto ndim                      = temp_shape.size();

  // If none of the dimensions are allowed to be partitioned, there's nothing the partitioner can
  // do.
  if (ndim == 0) {
    return {};
  }

  // Apparently captured structured bindings are only since C++20. Why on earth did the
  // committee not allow this in C++17????
  LEGATE_CPP_VERSION_TODO(20, "Use captured structured bindings instead");
  auto temp_result = [&](Point* shape_2, std::uint64_t volume_2) {
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

  auto result = Point::full(shape.dim(), 1);

  for (std::uint32_t idx = 0; idx < ndim; ++idx) {
    result[temp_dims[idx]] = temp_result[idx];
  }

  // TODO(wonchanl): The handling of minimum extent constraints here is rather simple as it goes
  // through the same partitioning heuristic and later gives up if the shape isn't big enough to
  // satisfy the constraints. This is to be in line with the current size-based heuristic. Plus, the
  // current use case for minimum-extent constraints is handling corner cases correctly and not
  // kernel efficiency. We may revisit this later if a true reduced degree of parallelism (i.e.,
  // using fewer GPUs than what's available in the scope) turns out desirable for performance
  // reasons.
  if (!restrictions.minimum_extents_satisfied_by(shape, result)) {
    return {};
  }

  return result;
}

Point PartitionManager::compute_tile_shape(const Point& extents, const Point& launch_shape)
{
  LEGATE_CHECK(extents.dim() == launch_shape.dim());

  auto tile_shape = extents;

  tile_shape += launch_shape;
  tile_shape -= 1;
  tile_shape /= launch_shape;

  return tile_shape;
}

}  // namespace legate::detail
