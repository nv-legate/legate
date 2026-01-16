/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/io/hdf5/detail/hdf5_partitioner.h>
#include <legate/io/hdf5/detail/hdf5_wrapper.h>

#include <vector>

namespace legate::io::hdf5::detail {

/**
 * @brief Partitioner for Virtual Dataset (VDS) HDF5 files.
 *
 * Analyzes VDS source files and determines optimal tiling based on
 * the block structure and source file layouts. The VDS must contain source files
 * of the same layout. The source files must have contiguous layout or chunked layout.
 * In case of chunked layout, the chunk shapes must be the same from all the source files.
 * in case of contiguous layout, the block shapes must be the same from all the source
 * files except the edge blocks.
 */
class VDSPartitioner final : public HDF5Partitioner {
 public:
  /**
   * @brief Partition a VDS into the given number of tiles and return the tile shape.
   *
   * Uses the dataset's chunk dimensions or block shapes as a basis for tiling with the following
   * logic:
   * - Non-slowest dimensions use chunk/block sizes (respecting shape boundaries)
   * - Slowest dimension is adjusted using a factor. This is calculated as the
   *   number of tiles needed if we use the chunk/block sizes for tiling divided by the number of
   * tiles requested.
   *   * Factor = tiles needed with chunk/block tiling / num_tiles
   *   * Adjusted tile size in slowest dimension = chunk size in slowest dimension * factor
   * Partitioning fails and returns std::nullopt if:
   * - Dataset has <=1 dimension or is empty
   * - num_tiles exceeds the slowest dimension size
   *
   * @param shape The shape of the dataset.
   * @param num_tiles The number of tiles to partition the dataset into.
   * @param vds_path The path to the VDS file.
   * @param dataset The dataset.
   *
   * @return The tile shape if partitioning is possible, otherwise std::nullopt.
   */
  [[nodiscard]] std::optional<legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>>
  partition_tile_shape(const Shape& shape,
                       std::size_t num_tiles,
                       const std::filesystem::path& vds_path,
                       const wrapper::HDF5DataSet& dataset) override;

 private:
  /**
   * @brief Collection of block shapes and positions from a virtual dataset.
   *
   * First element: block_shapes - Shape of each block (source file)
   * Second element: block_offsets - Offset of each block in virtual space
   */
  using VDSBlockInfo = std::pair<std::vector<legate::detail::SmallVector<hsize_t>>,
                                 std::vector<legate::detail::SmallVector<hsize_t>>>;

  /**
   * @brief Gather block shapes and positions from a virtual dataset (VDS).
   *
   * @param dataset The dataset.
   * @param shape The shape of the dataset.
   *
   * @return The block shapes and positions.
   */
  [[nodiscard]] VDSBlockInfo gather_contigous_block_info_(const wrapper::HDF5DataSet& dataset,
                                                          const Shape& shape);

  /**
   * @brief Get block shape if VDS has uniform tiling. Only the edge shapes are allowed to have
   * different sizes. The other shapes must be the same as the maximum shape. The source files
   * must have contiguous layout.
   *
   * @param dataset The dataset.
   * @param shape The shape of the dataset.
   *
   * @return The block shape if VDS has uniform tiling, otherwise std::nullopt.
   */
  [[nodiscard]] std::optional<legate::detail::SmallVector<hsize_t, LEGATE_MAX_DIM>>
  get_contigous_shape_if_uniform_(const wrapper::HDF5DataSet& dataset, const Shape& shape);

  /**
   * @brief Analyze source file layouts in a VDS. Check if all the source files use the same
   * layout.
   *
   * @param dcpl The dataset creation property list.
   * @param vds_path The path to the VDS file.
   *
   * @return The layout of the source files. H5D_CONTIGUOUS if all the source files are contiguous,
   * H5D_CHUNKED if all the source files are chunked, and H5D_LAYOUT_ERROR if the source files use
   * a different layout.
   */
  [[nodiscard]] H5D_layout_t get_source_layouts_(const wrapper::HDF5DataSetCreatePropertyList& dcpl,
                                                 const std::filesystem::path& vds_path);

  /**
   * @brief Check if all the chunk shapes are the same from all the source files with chunked
   * layout.
   *
   * @param dcpl The dataset creation property list.
   * @param vds_path The path to the VDS file.
   *
   * @return The chunk shape if all the chunk shapes are the same, otherwise std::nullopt.
   */
  [[nodiscard]] std::optional<legate::detail::SmallVector<hsize_t, LEGATE_MAX_DIM>>
  get_chunk_shape_info_if_uniform_(const wrapper::HDF5DataSetCreatePropertyList& dcpl,
                                   const std::filesystem::path& vds_path);
};

}  // namespace legate::io::hdf5::detail
