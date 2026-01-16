/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/io/hdf5/detail/partitioners/chunked_partitioner.h>

#include <legate/data/shape.h>
#include <legate/io/hdf5/detail/hdf5_wrapper.h>
#include <legate/utilities/abort.h>

#include <algorithm>

namespace legate::io::hdf5::detail {

std::optional<legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>>
ChunkedPartitioner::partition_tile_shape(const Shape& shape,
                                         std::size_t num_tiles,
                                         const std::filesystem::path& /*vds_path*/,
                                         const wrapper::HDF5DataSet& dataset)
{
  LEGATE_CHECK(num_tiles > 0);

  auto chunk_shape = dataset.get_create_plist().get_chunk_dims(shape.ndim());

  LEGATE_CHECK(chunk_shape.size() == shape.ndim());

  // if the dataset is not valid or the number of tiles requested is greater than
  // the size of the slowest dimension, then we cannot use the tiling strategy
  if (shape.ndim() <= 1 || shape.volume() == 0 || num_tiles > shape[0]) {
    return std::nullopt;
  }

  legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape{
    legate::detail::tags::size_tag, shape.ndim(), 0ULL};

  for (std::uint32_t i = 0; i < shape.ndim(); ++i) {
    tile_shape[i] = chunk_shape[i];
  }

  // tile along the slowest dimension
  const auto slowest_dim = 0;

  // Calculate tiles needed
  std::uint64_t tiles_needed = 1;
  for (std::uint32_t i = 0; i < shape.ndim(); ++i) {
    const auto tiles_in_dim =
      (static_cast<std::uint64_t>(shape[i]) + tile_shape[i] - 1) / tile_shape[i];
    tiles_needed *= tiles_in_dim;
  }

  // Calculate the adjustment factor for the slowest dimension
  // Factor > 1: fewer tiles (larger tile size)
  // Factor < 1: more tiles (smaller tile size)
  // Factor = 1: no adjustment (use chunk size)
  const double adjustment_factor =
    static_cast<double>(tiles_needed) / static_cast<double>(num_tiles);

  // Apply the factor to adjust the slowest dimension tile size
  const auto base_slowest_tile_size = static_cast<double>(tile_shape[slowest_dim]);
  const auto adjusted_tile_size =
    static_cast<std::uint64_t>(base_slowest_tile_size * adjustment_factor);

  // Clamp the adjusted tile size to valid bounds
  tile_shape[slowest_dim] = std::clamp(
    adjusted_tile_size, std::uint64_t{1}, static_cast<std::uint64_t>(shape[slowest_dim]));
  return tile_shape;
}

}  // namespace legate::io::hdf5::detail
