/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/io/hdf5/detail/hdf5_partitioner.h>

namespace legate::io::hdf5::detail {

/**
 * @brief Partitioner for contiguous HDF5 datasets.
 *
 * Tiles along the slowest dimension based on the number of requested tiles.
 */
class ContiguousPartitioner final : public HDF5Partitioner {
 public:
  /**
   * @brief Partition a contiguous dataset into the given number of tiles.
   * This is possible if the dataset has more than 1 dimension, is not empty, and the number of
   * tiles is less than the size of the slowest dimension.
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
