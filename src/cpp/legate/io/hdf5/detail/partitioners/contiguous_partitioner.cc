/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/io/hdf5/detail/partitioners/contiguous_partitioner.h>

#include <legate/data/shape.h>
#include <legate/io/hdf5/detail/hdf5_wrapper.h>

#include <algorithm>

namespace legate::io::hdf5::detail {

std::optional<legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>>
ContiguousPartitioner::partition_tile_shape(const Shape& shape,
                                            std::size_t num_tiles,
                                            const std::filesystem::path& /*vds_path*/,
                                            const wrapper::HDF5DataSet& /*dataset*/)
{
  if (shape.ndim() <= 1 || shape.volume() == 0 || num_tiles > shape[0]) {
    return std::nullopt;
  }

  const auto slowest_dim      = 0;
  const auto slowest_dim_size = static_cast<std::size_t>(shape[slowest_dim]);

  std::size_t tile_size_slowest = (slowest_dim_size + num_tiles - 1) / num_tiles;
  tile_size_slowest             = std::clamp(tile_size_slowest, std::size_t{1}, slowest_dim_size);

  // Create tile shape - only tile along slowest dimension, keep full size for others
  legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape{
    legate::detail::tags::size_tag, shape.ndim(), 0ULL};

  for (std::uint32_t i = 0; i < shape.ndim(); ++i) {
    tile_shape[i] = shape[i];
  }
  tile_shape[slowest_dim] = tile_size_slowest;

  return tile_shape;
}

}  // namespace legate::io::hdf5::detail
