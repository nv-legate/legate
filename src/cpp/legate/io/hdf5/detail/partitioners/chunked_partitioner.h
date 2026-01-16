/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/io/hdf5/detail/hdf5_partitioner.h>

namespace legate::io::hdf5::detail {

/**
 * @brief Partitioner for chunked HDF5 datasets.
 *
 * Uses the chunk shape from the dataset to determine optimal tiling.
 */
class ChunkedPartitioner final : public HDF5Partitioner {
 public:
  /**
   * @brief Partition a chunked dataset into the given number of tiles.
   *
   * Uses the dataset's chunk dimensions as a basis for tiling with the following logic:
   * - Non-slowest dimensions use chunk sizes (respecting shape boundaries)
   * - Slowest dimension is adjusted using a factor. This is calculated as the
   *   number of tiles needed if we use the chunk sizes for tiling divided by the number of tiles
   * requested.
   *   * Factor = tiles needed with chunk tiling / num_tiles
   *   * Adjusted tile size in slowest dimension = chunk size in slowest dimension * factor
   * Partitioning fails and returns std::nullopt if:
   * - Dataset has <=1 dimension or is empty
   * - num_tiles exceeds the slowest dimension size
   *
   * @return The tile shape if partitioning is possible, otherwise std::nullopt.
   */
  [[nodiscard]] std::optional<legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>>
  partition_tile_shape(const Shape& shape,
                       std::size_t num_tiles,
                       const std::filesystem::path& vds_path,
                       const wrapper::HDF5DataSet& dataset) override;
};

}  // namespace legate::io::hdf5::detail
