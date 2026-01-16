/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/io/hdf5/detail/hdf5_partitioner.h>

#include <legate/data/shape.h>
#include <legate/io/hdf5/detail/hdf5_wrapper.h>
#include <legate/io/hdf5/detail/partitioners/chunked_partitioner.h>
#include <legate/io/hdf5/detail/partitioners/contiguous_partitioner.h>
#include <legate/io/hdf5/detail/partitioners/vds_partitioner.h>
#include <legate/utilities/abort.h>

namespace legate::io::hdf5::detail {

std::optional<legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>> get_partition_tile_shape(
  const Shape& shape,
  std::size_t num_tiles,
  const std::filesystem::path& vds_path,
  const wrapper::HDF5DataSet& dataset)
{
  switch (dataset.get_layout()) {
    case H5D_VIRTUAL:
      return VDSPartitioner{}.partition_tile_shape(shape, num_tiles, vds_path, dataset);
    case H5D_CHUNKED:
      return ChunkedPartitioner{}.partition_tile_shape(shape, num_tiles, vds_path, dataset);
    case H5D_CONTIGUOUS:
      return ContiguousPartitioner{}.partition_tile_shape(shape, num_tiles, vds_path, dataset);
    case H5D_LAYOUT_ERROR: [[fallthrough]];
    case H5D_COMPACT: [[fallthrough]];
    case H5D_NLAYOUTS: return std::nullopt;
  }
  LEGATE_ABORT("Unsupported dataset layout");
}

}  // namespace legate::io::hdf5::detail
