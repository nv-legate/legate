/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/io/hdf5/detail/partitioners/vds_partitioner.h>

#include <legate/data/shape.h>
#include <legate/utilities/abort.h>

#include <algorithm>

namespace legate::io::hdf5::detail {

std::optional<legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>>
VDSPartitioner::partition_tile_shape(const Shape& shape,
                                     std::size_t num_tiles,
                                     const std::filesystem::path& vds_path,
                                     const wrapper::HDF5DataSet& dataset)
{
  if (shape.ndim() <= 1 || shape.volume() == 0 || shape[0] < num_tiles) {
    return std::nullopt;
  }

  auto dcpl = dataset.get_create_plist();

  if (dcpl.virtual_count() == 0) {
    return std::nullopt;
  }

  const auto layout = get_source_layouts_(dcpl, vds_path);

  if (layout == H5D_LAYOUT_ERROR) {
    return std::nullopt;
  }

  legate::detail::SmallVector<hsize_t, LEGATE_MAX_DIM> standard_shape{};

  if (layout == H5D_CONTIGUOUS) {
    auto ret = get_contigous_shape_if_uniform_(dataset, shape);

    // if blocks are not uniform then we cannot use the tiling strategy
    if (!ret) {
      return std::nullopt;
    }

    standard_shape = *ret;
  }

  if (layout == H5D_CHUNKED) {
    // check if all the chunk shapes are the same from all the source files
    auto chunk_shape = get_chunk_shape_info_if_uniform_(dcpl, vds_path);

    if (!chunk_shape) {
      return std::nullopt;
    }

    standard_shape = *chunk_shape;
  }

  const auto slowest_dim = 0;
  legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape{
    legate::detail::tags::size_tag, shape.ndim(), 0ULL};

  for (std::uint32_t i = 0; i < shape.ndim(); ++i) {
    tile_shape[i] = standard_shape[i];
  }

  // Calculate tiles needed in non-slowest dimensions
  std::uint64_t tiles_needed = 1;
  for (std::uint32_t i = 0; i < shape.ndim(); ++i) {
    const auto tiles_in_dim = (shape[i] + tile_shape[i] - 1) / tile_shape[i];
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
  tile_shape[slowest_dim] = std::clamp(adjusted_tile_size, std::uint64_t{1}, shape[slowest_dim]);
  return tile_shape;
}

VDSPartitioner::VDSBlockInfo VDSPartitioner::gather_contigous_block_info_(
  const wrapper::HDF5DataSet& dataset, const Shape& shape)
{
  std::vector<legate::detail::SmallVector<hsize_t>> block_shapes;
  std::vector<legate::detail::SmallVector<hsize_t>> block_offsets;

  // Get the dataset creation property list
  auto dcpl                = dataset.get_create_plist();
  const auto virtual_count = dcpl.virtual_count();

  for (std::size_t i = 0; i < virtual_count; ++i) {
    // Get virtual dataspace - this gives us where the block is placed and its size
    auto vds_space       = wrapper::HDF5VirtualSpace{dcpl.hid(), i};
    auto [block, offset] = vds_space.get_select_bounds(shape.ndim());

    block_shapes.emplace_back(std::move(block));
    block_offsets.emplace_back(std::move(offset));
  }

  return {std::move(block_shapes), std::move(block_offsets)};
}

std::optional<legate::detail::SmallVector<hsize_t, LEGATE_MAX_DIM>>
VDSPartitioner::get_contigous_shape_if_uniform_(const wrapper::HDF5DataSet& dataset,
                                                const Shape& shape)
{
  const auto& [block_shapes, block_offsets] = gather_contigous_block_info_(dataset, shape);

  if (block_shapes.empty()) {
    return std::nullopt;
  }

  legate::detail::SmallVector<hsize_t, LEGATE_MAX_DIM> reference_shape{
    legate::detail::tags::size_tag, shape.ndim(), 0ULL};

  // Helper lambda to check if a block is at any edge
  auto is_block_at_edge = [&](const auto& block_shape, const auto& block_offset) -> bool {
    for (std::uint32_t dim = 0; dim < shape.ndim(); ++dim) {
      if (block_offset[dim] + block_shape[dim] >= shape[dim]) {
        return true;
      }
    }
    return false;
  };

  // Helper lambda to check if block has zero offset
  auto has_zero_offset = [&](const auto& block_offset) -> bool {
    for (std::uint32_t dim = 0; dim < shape.ndim(); ++dim) {
      if (block_offset[dim] != 0) {
        return false;
      }
    }
    return true;
  };

  bool found_reference_block = false;

  // ind reference block (prefer non-edge, fallback to zero-offset)
  for (std::size_t i = 0; i < block_shapes.size(); ++i) {
    const auto& block_shape  = block_shapes[i];
    const auto& block_offset = block_offsets[i];

    if (!is_block_at_edge(block_shape, block_offset)) {
      for (std::uint32_t dim = 0; dim < shape.ndim(); ++dim) {
        reference_shape[dim] = block_shape[dim];
      }
      found_reference_block = true;
      break;
    }

    if (has_zero_offset(block_offset)) {
      for (std::uint32_t dim = 0; dim < shape.ndim(); ++dim) {
        reference_shape[dim] = block_shape[dim];
      }
      found_reference_block = true;
    }
  }

  if (!found_reference_block) {
    return std::nullopt;
  }

  for (std::size_t i = 0; i < block_shapes.size(); ++i) {
    const auto& block_shape  = block_shapes[i];
    const auto& block_offset = block_offsets[i];

    if (!is_block_at_edge(block_shape, block_offset)) {
      for (std::uint32_t dim = 0; dim < shape.ndim(); ++dim) {
        if (block_shape[dim] != reference_shape[dim]) {
          return std::nullopt;
        }
      }
    }
  }

  return reference_shape;
}

H5D_layout_t VDSPartitioner::get_source_layouts_(const wrapper::HDF5DataSetCreatePropertyList& dcpl,
                                                 const std::filesystem::path& vds_path)
{
  const auto virtual_count = dcpl.virtual_count();

  if (virtual_count == 0) {
    return H5D_LAYOUT_ERROR;
  }

  const auto vds_dir  = vds_path.parent_path();
  bool has_contiguous = false;
  bool has_chunked    = false;

  for (std::size_t i = 0; i < virtual_count; ++i) {
    const auto src_filename = dcpl.virtual_filename(i);
    const auto src_dsetname = dcpl.virtual_dsetname(i);

    // Resolve relative paths relative to VDS file location
    const std::filesystem::path src_path = [&]() {
      auto path = std::filesystem::path{src_filename};

      if (path.is_relative()) {
        path = vds_dir / path;
      }
      return path;
    }();

    const wrapper::HDF5File src_file{src_path.native(), wrapper::HDF5File::OpenMode::READ_ONLY};
    const auto layout = src_file.data_set(src_dsetname).get_layout();

    has_contiguous = layout == H5D_CONTIGUOUS;
    has_chunked    = layout == H5D_CHUNKED;

    if (has_contiguous && has_chunked) {
      return H5D_LAYOUT_ERROR;
    }
  }

  return has_contiguous ? H5D_CONTIGUOUS : H5D_CHUNKED;
}

std::optional<legate::detail::SmallVector<hsize_t, LEGATE_MAX_DIM>>
VDSPartitioner::get_chunk_shape_info_if_uniform_(const wrapper::HDF5DataSetCreatePropertyList& dcpl,
                                                 const std::filesystem::path& vds_path)
{
  const auto virtual_count = dcpl.virtual_count();

  LEGATE_CHECK(virtual_count > 0);

  std::vector<legate::detail::SmallVector<hsize_t, LEGATE_MAX_DIM>> chunk_dims;
  const auto vds_dir = vds_path.parent_path();

  for (std::size_t i = 0; i < virtual_count; ++i) {
    const auto src_filename = dcpl.virtual_filename(i);
    const auto src_dsetname = dcpl.virtual_dsetname(i);

    // Resolve relative paths relative to VDS file location
    const std::filesystem::path src_path = [&]() {
      auto path = std::filesystem::path{src_filename};
      if (path.is_relative()) {
        path = vds_dir / path;
      }
      return path;
    }();

    const wrapper::HDF5File src_file{src_path.native(), wrapper::HDF5File::OpenMode::READ_ONLY};
    const auto src_dataset = src_file.data_set(src_dsetname);
    auto chunk_dim =
      src_dataset.get_create_plist().get_chunk_dims(src_dataset.data_space().extents().size());
    const auto&& extents = src_dataset.data_space().extents();

    chunk_dims.emplace_back(std::move(chunk_dim));
  }

  // check if all the chunk shapes are the same from all the source files
  if (!std::all_of(chunk_dims.begin(), chunk_dims.end(), [&](const auto& chunk_dim) {
        return chunk_dims[0].size() == chunk_dim.size() &&
               std::equal(chunk_dims[0].begin(), chunk_dims[0].end(), chunk_dim.begin());
      })) {
    return {};
  }

  return chunk_dims[0];
}

}  // namespace legate::io::hdf5::detail
