/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/small_vector.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>

namespace legate {

class Shape;

}  // namespace legate

namespace legate::io::hdf5::detail::wrapper {

class HDF5DataSet;

}  // namespace legate::io::hdf5::detail::wrapper

namespace legate::io::hdf5::detail {

/** Interface class for partitioning HDF5 datasets. */
class HDF5Partitioner {
 public:
  /**
   * @brief Destructor.
   */
  virtual ~HDF5Partitioner() = default;

  /**
   * @brief Partition the dataset into the given number of tiles.
   *
   * @param shape The shape of the dataset.
   * @param num_tiles The number of tiles to partition the dataset into.
   * @param vds_path The path to the VDS file.
   * @param dataset The dataset.
   *
   * @return The tile shape if partitioning is possible, otherwise std::nullopt.
   */
  [[nodiscard]] virtual std::optional<legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>>
  partition_tile_shape(const Shape& shape,
                       std::size_t num_tiles,
                       const std::filesystem::path& vds_path,
                       const wrapper::HDF5DataSet& dataset) = 0;
};

/**
 * @brief Get the partition tile shape for a given dataset.
 *
 * Determines the optimal tile shape based on the dataset layout (chunked, contiguous, or VDS).
 *
 * @param shape The shape of the dataset.
 * @param num_tiles The number of tiles to partition the dataset into.
 * @param vds_path The path to the VDS file.
 * @param dataset The dataset.
 *
 * @return The tile shape if partitioning is possible, otherwise std::nullopt.
 */
[[nodiscard]] std::optional<legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>>
get_partition_tile_shape(const Shape& shape,
                         std::size_t num_tiles,
                         const std::filesystem::path& vds_path,
                         const wrapper::HDF5DataSet& dataset);

}  // namespace legate::io::hdf5::detail
