/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/io/hdf5/detail/hdf5_partitioner.h>
#include <legate/io/hdf5/detail/hdf5_wrapper.h>
#include <legate/io/hdf5/interface.h>

#include <H5Fpublic.h>
#include <H5Gpublic.h>
#include <H5Ppublic.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <filesystem>
#include <utilities/utilities.h>

namespace test_io_hdf5_tiling {

namespace {

/**
 * @brief Helper function to create HDF5 file with sequential float data
 *
 * @param file_path The path to the HDF5 file.
 * @param dataset_name The name of the dataset.
 * @param dims The dimensions of the dataset.
 * @param chunk_dims Optional chunk dimensions. If provided, creates a chunked dataset;
 *                   otherwise creates a contiguous dataset.
 */
template <std::size_t NDIM>
void create_hdf5_file_with_sequential_data(const std::filesystem::path& file_path,
                                           const std::string& dataset_name,
                                           const std::array<hsize_t, NDIM>& dims,
                                           const std::array<hsize_t, NDIM>* chunk_dims = nullptr)
{
  const auto file = H5Fcreate(file_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  ASSERT_GE(file, 0);

  const auto space = H5Screate_simple(dims.size(), dims.data(), nullptr);

  ASSERT_GE(space, 0);

  // Create dataset creation property list (with optional chunking)
  hid_t dcpl = H5P_DEFAULT;

  if (chunk_dims != nullptr) {
    dcpl = H5Pcreate(H5P_DATASET_CREATE);

    ASSERT_GE(dcpl, 0);
    ASSERT_GE(H5Pset_chunk(dcpl, NDIM, chunk_dims->data()), 0);
  }

  const auto dset =
    H5Dcreate(file, dataset_name.c_str(), H5T_IEEE_F32LE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);

  ASSERT_GE(dset, 0);

  std::size_t total_size = 1;

  for (auto dim : dims) {
    total_size *= dim;
  }

  auto data = std::vector<float>(total_size);

  // Fill with sequential indices: 0, 1, 2, ..., TOTAL_SIZE-1
  for (std::size_t i = 0; i < total_size; ++i) {
    data[i] = static_cast<float>(i);
  }

  ASSERT_GE(H5Dwrite(dset, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()), 0);

  // Clean up HDF5 resources
  if (dcpl != H5P_DEFAULT) {
    ASSERT_GE(H5Pclose(dcpl), 0);
  }
  ASSERT_GE(H5Sclose(space), 0);
  ASSERT_GE(H5Dclose(dset), 0);
  ASSERT_GE(H5Fclose(file), 0);
}

/**
 * @brief Helper function to create a Virtual Dataset (VDS) with sequential float data
 *
 * Creates multiple source HDF5 files tiled along the slowest dimension, then combines
 * them into a single VDS file.
 *
 * @param vds_file_path The path to the main VDS file.
 * @param source_dir The directory to store source files.
 * @param dataset_name The name of the dataset.
 * @param dims The total dimensions of the virtual dataset.
 * @param num_tiles Number of tiles along the slowest dimension.
 * @param chunk_dims Optional chunk dimensions. If provided, source files use chunked storage;
 *                   otherwise they use contiguous storage.
 */
template <std::size_t NDIM>
void create_vds_with_sequential_data(const std::filesystem::path& vds_file_path,
                                     const std::filesystem::path& source_dir,
                                     const std::string& dataset_name,
                                     const std::array<hsize_t, NDIM>& dims,
                                     std::size_t num_files,
                                     const std::array<hsize_t, NDIM>* chunk_dims = nullptr)
{
  static_assert(NDIM >= 1, "VDS requires at least 1 dimension");
  ASSERT_TRUE(std::filesystem::create_directories(source_dir));

  const hsize_t total_slowest_dim = dims[0];
  const hsize_t tile_size         = (total_slowest_dim + num_files - 1) / num_files;

  // Calculate the size of one "slice" (product of all dims except the slowest)
  std::size_t slice_size = 1;

  for (std::size_t d = 1; d < NDIM; ++d) {
    slice_size *= dims[d];
  }

  // Create source files and track their paths
  std::vector<std::filesystem::path> source_paths;

  for (std::size_t tile_idx = 0; tile_idx < num_files; ++tile_idx) {
    const hsize_t start_idx     = tile_idx * tile_size;
    const hsize_t end_idx       = std::min(start_idx + tile_size, total_slowest_dim);
    const hsize_t this_tile_dim = end_idx - start_idx;

    if (this_tile_dim == 0) {
      break;
    }

    std::array<hsize_t, NDIM> src_dims = dims;
    src_dims[0]                        = this_tile_dim;

    const auto src_file_path = source_dir / ("tile_" + std::to_string(tile_idx) + ".h5");
    source_paths.push_back(src_file_path);

    // Create source file
    const auto src_file = H5Fcreate(src_file_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    ASSERT_GE(src_file, 0);

    const auto src_space = H5Screate_simple(NDIM, src_dims.data(), nullptr);

    ASSERT_GE(src_space, 0);

    // Create dataset creation property list (with optional chunking)
    hid_t src_dcpl = H5P_DEFAULT;
    if (chunk_dims != nullptr) {
      // Clamp chunk dimensions to not exceed source file dimensions
      std::array<hsize_t, NDIM> clamped_chunk_dims;

      for (std::size_t d = 0; d < NDIM; ++d) {
        clamped_chunk_dims[d] = std::min((*chunk_dims)[d], src_dims[d]);
      }

      src_dcpl = H5Pcreate(H5P_DATASET_CREATE);
      ASSERT_GE(src_dcpl, 0);
      ASSERT_GE(H5Pset_chunk(src_dcpl, NDIM, clamped_chunk_dims.data()), 0);
    }

    const auto src_dset = H5Dcreate(src_file,
                                    dataset_name.c_str(),
                                    H5T_IEEE_F32LE,
                                    src_space,
                                    H5P_DEFAULT,
                                    src_dcpl,
                                    H5P_DEFAULT);
    ASSERT_GE(src_dset, 0);

    // Fill source data with sequential values based on global position
    const std::size_t src_total = this_tile_dim * slice_size;
    std::vector<float> src_data(src_total);

    for (hsize_t local_slow = 0; local_slow < this_tile_dim; ++local_slow) {
      const hsize_t global_slow = start_idx + local_slow;

      for (std::size_t slice_idx = 0; slice_idx < slice_size; ++slice_idx) {
        const std::size_t global_index = (global_slow * slice_size) + slice_idx;
        const std::size_t local_index  = (local_slow * slice_size) + slice_idx;
        src_data[local_index]          = static_cast<float>(global_index);
      }
    }

    ASSERT_GE(H5Dwrite(src_dset, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, src_data.data()),
              0);

    // Clean up source file resources
    if (src_dcpl != H5P_DEFAULT) {
      ASSERT_GE(H5Pclose(src_dcpl), 0);
    }
    ASSERT_GE(H5Sclose(src_space), 0);
    ASSERT_GE(H5Dclose(src_dset), 0);
    ASSERT_GE(H5Fclose(src_file), 0);
  }

  // Create the VDS file
  const auto vds_file = H5Fcreate(vds_file_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  ASSERT_GE(vds_file, 0);

  const auto vds_space = H5Screate_simple(NDIM, dims.data(), nullptr);

  ASSERT_GE(vds_space, 0);

  // Create dataset creation property list for VDS
  const auto dcpl = H5Pcreate(H5P_DATASET_CREATE);

  ASSERT_GE(dcpl, 0);

  // Add virtual mappings for each source file
  for (std::size_t tile_idx = 0; tile_idx < source_paths.size(); ++tile_idx) {
    const hsize_t start_idx     = tile_idx * tile_size;
    const hsize_t end_idx       = std::min(start_idx + tile_size, total_slowest_dim);
    const hsize_t this_tile_dim = end_idx - start_idx;

    std::array<hsize_t, NDIM> src_dims = dims;
    std::array<hsize_t, NDIM> vds_start{};
    std::array<hsize_t, NDIM> vds_count{};

    src_dims[0]  = this_tile_dim;
    vds_start[0] = start_idx;
    vds_count.fill(1);

    std::array<hsize_t, NDIM> vds_block = src_dims;

    ASSERT_GE(
      H5Sselect_hyperslab(
        vds_space, H5S_SELECT_SET, vds_start.data(), nullptr, vds_count.data(), vds_block.data()),
      0);

    const auto src_space = H5Screate_simple(NDIM, src_dims.data(), nullptr);

    ASSERT_GE(src_space, 0);
    ASSERT_GE(H5Sselect_all(src_space), 0);

    ASSERT_GE(H5Pset_virtual(
                dcpl, vds_space, source_paths[tile_idx].c_str(), dataset_name.c_str(), src_space),
              0);

    ASSERT_GE(H5Sclose(src_space), 0);
  }

  const auto vds_dset = H5Dcreate(
    vds_file, dataset_name.c_str(), H5T_IEEE_F32LE, vds_space, H5P_DEFAULT, dcpl, H5P_DEFAULT);

  ASSERT_GE(vds_dset, 0);
  ASSERT_GE(H5Pclose(dcpl), 0);
  ASSERT_GE(H5Sclose(vds_space), 0);
  ASSERT_GE(H5Dclose(vds_dset), 0);
  ASSERT_GE(H5Fclose(vds_file), 0);
}

/**
 * @brief Helper to open an HDF5 file and compute partition tile shape
 */
template <std::size_t NDIM>
auto get_tile_shape_for_file(const std::filesystem::path& file_path,
                             const std::string& dataset_name,
                             const std::array<hsize_t, NDIM>& dims,
                             int num_tiles)
{
  const auto file = legate::io::hdf5::detail::wrapper::HDF5File{
    file_path.native(), legate::io::hdf5::detail::wrapper::HDF5File::OpenMode::READ_ONLY};
  auto dataset = file.data_set(dataset_name);
  auto shape   = legate::Shape{dims};

  return legate::io::hdf5::detail::get_partition_tile_shape(shape, num_tiles, file_path, dataset);
}

class IOHDF5TilingUnit : public ::testing::Test {
 public:
  void SetUp() override
  {
    Test::SetUp();
    ASSERT_NO_THROW(std::filesystem::create_directories(base_path));
  }

  void TearDown() override
  {
    Test::TearDown();
    ASSERT_NO_THROW(static_cast<void>(std::filesystem::remove_all(base_path)));
  }

  // NOLINTNEXTLINE(cert-err58-cpp)
  static inline auto base_path = std::filesystem::temp_directory_path() / "legate";
};

}  // namespace

class IOHDF5TilingThreeDContiguousParam : public IOHDF5TilingUnit,
                                          public ::testing::WithParamInterface<int> {};

TEST_P(IOHDF5TilingThreeDContiguousParam, ThreeDimensionalContiguous)
{
  constexpr auto X       = 400;
  constexpr auto Y       = 300;
  constexpr auto Z       = 200;
  constexpr auto DATASET = "/three_dimensional";
  constexpr auto DIM     = 3;
  const auto file_path   = base_path / "three_d_contiguous.h5";

  constexpr auto dims = std::array<hsize_t, DIM>{X, Y, Z};
  create_hdf5_file_with_sequential_data(file_path, DATASET, dims);

  const auto num_tiles  = GetParam();
  const auto tile_shape = get_tile_shape_for_file(file_path, DATASET, dims, num_tiles);

  ASSERT_TRUE(tile_shape.has_value());

  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  const auto& ts = *tile_shape;
  ASSERT_EQ(ts.size(), DIM);
  ASSERT_EQ(ts[0], X / num_tiles);
  ASSERT_EQ(ts[1], Y);
  ASSERT_EQ(ts[2], Z);
}

INSTANTIATE_TEST_SUITE_P(NumTiles,
                         IOHDF5TilingThreeDContiguousParam,
                         ::testing::Values(10, 20, 50));

class IOHDF5TilingTwoDContiguousParam : public IOHDF5TilingUnit,
                                        public ::testing::WithParamInterface<int> {};

TEST_P(IOHDF5TilingTwoDContiguousParam, TwoDimensionalContiguous)
{
  constexpr auto X       = 100;
  constexpr auto Y       = 500;
  constexpr auto DATASET = "/two_dimensional";
  constexpr auto DIM     = 2;
  const auto file_path   = base_path / "two_d_contiguous.h5";

  constexpr auto dims = std::array<hsize_t, DIM>{X, Y};

  create_hdf5_file_with_sequential_data(file_path, DATASET, dims);

  const auto num_tiles  = GetParam();
  const auto tile_shape = get_tile_shape_for_file(file_path, DATASET, dims, num_tiles);

  ASSERT_TRUE(tile_shape.has_value());

  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  const auto& ts = *tile_shape;
  ASSERT_EQ(ts.size(), DIM);
  ASSERT_EQ(ts[0], X / num_tiles);
  ASSERT_EQ(ts[1], Y);
}

INSTANTIATE_TEST_SUITE_P(NumTiles, IOHDF5TilingTwoDContiguousParam, ::testing::Values(25, 50));

TEST_F(IOHDF5TilingUnit, TwoDimensionalContiguousCannotPartition)
{
  constexpr auto X       = 100;
  constexpr auto Y       = 500;
  constexpr auto DATASET = "/two_dimensional";
  constexpr auto DIM     = 2;
  const auto file_path   = base_path / "two_d_contiguous_cannot_partition.h5";

  constexpr auto dims = std::array<hsize_t, DIM>{X, Y};

  create_hdf5_file_with_sequential_data(file_path, DATASET, dims);

  const auto num_tiles = X + 1;
  ASSERT_FALSE(get_tile_shape_for_file(file_path, DATASET, dims, num_tiles).has_value());
}

class IOHDF5TilingThreeDChunkedParam : public IOHDF5TilingUnit,
                                       public ::testing::WithParamInterface<int> {};

TEST_P(IOHDF5TilingThreeDChunkedParam, ThreeDimensionalChunked)
{
  constexpr auto X = 400;
  constexpr auto Y = 300;
  constexpr auto Z = 200;

  constexpr auto X_CHUNK = 50;
  constexpr auto Y_CHUNK = 100;
  constexpr auto Z_CHUNK = 100;

  constexpr auto DATASET = "/three_dimensional";
  constexpr auto DIM     = 3;
  const auto file_path   = base_path / "three_d_chunked.h5";

  constexpr auto dims       = std::array<hsize_t, DIM>{X, Y, Z};
  constexpr auto chunk_dims = std::array<hsize_t, DIM>{X_CHUNK, Y_CHUNK, Z_CHUNK};

  create_hdf5_file_with_sequential_data(file_path, DATASET, dims, &chunk_dims);

  const auto num_tiles  = GetParam();
  const auto tile_shape = get_tile_shape_for_file(file_path, DATASET, dims, num_tiles);
  ASSERT_TRUE(tile_shape.has_value());

  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  const auto& ts = *tile_shape;

  // With our new factor-based algorithm:
  // tiles_needed = ceil(400/50) * ceil(300/100) * ceil(200/100) = 8 * 3 * 2 = 48
  // factor = tiles_needed / num_tiles
  // adjusted_tile_size = 50 * factor
  const auto tiles_needed =
    ((X + X_CHUNK - 1) / X_CHUNK) * ((Y + Y_CHUNK - 1) / Y_CHUNK) * ((Z + Z_CHUNK - 1) / Z_CHUNK);
  const double factor = static_cast<double>(tiles_needed) / static_cast<double>(num_tiles);
  const auto expected_slowest_tile =
    std::clamp(static_cast<std::uint64_t>(static_cast<double>(X_CHUNK) * factor),
               std::uint64_t{1},
               static_cast<std::uint64_t>(X));

  ASSERT_EQ(ts.size(), DIM);
  ASSERT_EQ(ts[0], expected_slowest_tile);
  ASSERT_EQ(ts[1], Y_CHUNK);
  ASSERT_EQ(ts[2], Z_CHUNK);
}

INSTANTIATE_TEST_SUITE_P(NumTiles, IOHDF5TilingThreeDChunkedParam, ::testing::Values(48, 96));

TEST_F(IOHDF5TilingUnit, ThreeDimensionalChunkedCannotPartition)
{
  constexpr auto X = 400;
  constexpr auto Y = 300;
  constexpr auto Z = 200;

  constexpr auto X_CHUNK = 50;
  constexpr auto Y_CHUNK = 100;
  constexpr auto Z_CHUNK = 100;

  constexpr auto DATASET = "/three_dimensional";
  constexpr auto DIM     = 3;
  const auto file_path   = base_path / "three_d_chunked_cannot_partition.h5";

  constexpr auto dims       = std::array<hsize_t, DIM>{X, Y, Z};
  constexpr auto chunk_dims = std::array<hsize_t, DIM>{X_CHUNK, Y_CHUNK, Z_CHUNK};

  create_hdf5_file_with_sequential_data(file_path, DATASET, dims, &chunk_dims);

  ASSERT_FALSE(get_tile_shape_for_file(file_path, DATASET, dims, X + 1).has_value());
}

class IOHDF5TilingTwoDVDSContiguousParam : public IOHDF5TilingUnit,
                                           public ::testing::WithParamInterface<int> {};

TEST_P(IOHDF5TilingTwoDVDSContiguousParam, TwoDimensionalVDSContiguous)
{
  constexpr auto X         = 100;
  constexpr auto Y         = 500;
  constexpr auto DIM       = 2;
  constexpr auto NUM_FILES = 10;
  constexpr auto DATASET   = "/two_dimensional";
  const auto vds_file_path = base_path / "two_d_vds_contiguous.h5";
  const auto source_dir    = base_path / "two_d_vds_contiguous_source";

  constexpr auto dims = std::array<hsize_t, DIM>{X, Y};

  create_vds_with_sequential_data(vds_file_path, source_dir, DATASET, dims, NUM_FILES);

  const auto num_tiles  = GetParam();
  const auto tile_shape = get_tile_shape_for_file(vds_file_path, DATASET, dims, num_tiles);

  ASSERT_TRUE(tile_shape.has_value());

  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  const auto& ts = *tile_shape;
  ASSERT_EQ(ts.size(), DIM);

  // For contiguous VDS, block shape is [X/NUM_FILES, Y] = [10, 500]
  // tiles_needed = ceil(100/10) * ceil(500/500) = 10 * 1 = 10
  // adjustment_factor = 10 / num_tiles
  // adjusted_tile_size = 10 * adjustment_factor
  const auto block_x      = X / NUM_FILES;                                      // 100/10 = 10
  const auto tiles_needed = ((X + block_x - 1) / block_x) * ((Y + Y - 1) / Y);  // 10 * 1 = 10
  const double factor     = static_cast<double>(tiles_needed) / static_cast<double>(num_tiles);
  const auto expected_slowest_tile =
    std::clamp(static_cast<int>(static_cast<double>(block_x) * factor), 1, X);

  ASSERT_EQ(ts[0], expected_slowest_tile);
  ASSERT_EQ(ts[1], Y);
}

INSTANTIATE_TEST_SUITE_P(NumTiles,
                         IOHDF5TilingTwoDVDSContiguousParam,
                         ::testing::Values(4, 10, 20, 50));

class IOHDF5TilingTwoDVDSChunkedParam : public IOHDF5TilingUnit,
                                        public ::testing::WithParamInterface<int> {};

TEST_P(IOHDF5TilingTwoDVDSChunkedParam, TwoDimensionalVDSChunked)
{
  constexpr auto X       = 100;
  constexpr auto Y       = 500;
  constexpr auto X_CHUNK = 10;
  constexpr auto Y_CHUNK = 20;

  constexpr auto NUM_FILES = 10;
  constexpr auto DIM       = 2;
  constexpr auto DATASET   = "/two_dimensional";
  const auto vds_file_path = base_path / "two_d_vds_chunked.h5";
  const auto source_dir    = base_path / "two_d_vds_chunked_source";

  constexpr auto dims       = std::array<hsize_t, DIM>{X, Y};
  constexpr auto chunk_dims = std::array<hsize_t, DIM>{X_CHUNK, Y_CHUNK};

  create_vds_with_sequential_data(vds_file_path, source_dir, DATASET, dims, NUM_FILES, &chunk_dims);

  const auto num_tiles  = GetParam();
  const auto tile_shape = get_tile_shape_for_file(vds_file_path, DATASET, dims, num_tiles);

  ASSERT_TRUE(tile_shape.has_value());

  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  const auto& ts = *tile_shape;
  ASSERT_EQ(ts.size(), DIM);

  // tiles_needed = ceil(100/10) * ceil(500/20) = 10 * 25 = 250
  // adjustment_factor = 250 / num_tiles
  // adjusted_tile_size = 10 * adjustment_factor
  const auto tiles_needed =
    ((X + X_CHUNK - 1) / X_CHUNK) * ((Y + Y_CHUNK - 1) / Y_CHUNK);  // 10 * 25 = 250
  const double factor = static_cast<double>(tiles_needed) / static_cast<double>(num_tiles);
  const auto expected_slowest_tile =
    std::clamp(static_cast<int>(static_cast<double>(X_CHUNK) * factor), 1, X);

  ASSERT_EQ(ts[0], expected_slowest_tile);
  ASSERT_EQ(ts[1], Y_CHUNK);
}

INSTANTIATE_TEST_SUITE_P(NumTiles, IOHDF5TilingTwoDVDSChunkedParam, ::testing::Values(10, 20, 50));

TEST_F(IOHDF5TilingUnit, TwoDimensionalVDSChunkedCannotPartition)
{
  constexpr auto X       = 100;
  constexpr auto Y       = 500;
  constexpr auto X_CHUNK = 10;
  constexpr auto Y_CHUNK = 20;

  constexpr auto NUM_FILES = 10;
  constexpr auto DIM       = 2;
  constexpr auto DATASET   = "/two_dimensional";
  const auto vds_file_path = base_path / "two_d_vds_chunked_cannot_partition.h5";
  const auto source_dir    = base_path / "two_d_vds_chunked_cannot_partition_source";

  constexpr auto dims       = std::array<hsize_t, DIM>{X, Y};
  constexpr auto chunk_dims = std::array<hsize_t, DIM>{X_CHUNK, Y_CHUNK};

  create_vds_with_sequential_data(vds_file_path, source_dir, DATASET, dims, NUM_FILES, &chunk_dims);

  const auto num_tiles  = X + 1;
  const auto tile_shape = get_tile_shape_for_file(vds_file_path, DATASET, dims, num_tiles);

  ASSERT_FALSE(tile_shape.has_value());
}

TEST_F(IOHDF5TilingUnit, ChunkedPartitionerSmallChunks)
{
  // Test case where we have many small chunks and fewer tiles
  constexpr auto X = 200;
  constexpr auto Y = 100;
  constexpr auto Z = 50;

  constexpr auto X_CHUNK = 10;  // 200/10 = 20 chunks in slowest dimension
  constexpr auto Y_CHUNK = 50;  // 100/50 = 2 chunks, but we take min(50, 100) = 50
  constexpr auto Z_CHUNK = 25;  // 50/25 = 2 chunks, but we take min(25, 50) = 25

  constexpr auto DATASET = "/small_chunks";
  constexpr auto DIM     = 3;
  const auto file_path   = base_path / "small_chunks.h5";

  constexpr auto dims       = std::array<hsize_t, DIM>{X, Y, Z};
  constexpr auto chunk_dims = std::array<hsize_t, DIM>{X_CHUNK, Y_CHUNK, Z_CHUNK};

  create_hdf5_file_with_sequential_data(file_path, DATASET, dims, &chunk_dims);

  const auto num_tiles  = 5;
  const auto tile_shape = get_tile_shape_for_file(file_path, DATASET, dims, num_tiles);
  ASSERT_TRUE(tile_shape.has_value());

  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  const auto& ts = *tile_shape;

  ASSERT_EQ(ts.size(), DIM);

  // - tiles_needed = ceil(200/10) * ceil(100/50) * ceil(50/25) = 20 * 2 * 2 = 80
  // - adjustment_factor = 80/5 = 16
  // - Adjusted slowest tile size = 10 * 16 = 160
  ASSERT_EQ(ts[0], 160);
  ASSERT_EQ(ts[1], Y_CHUNK);
  ASSERT_EQ(ts[2], Z_CHUNK);
}

TEST_F(IOHDF5TilingUnit, ChunkedPartitionerFactorBasedAdjustment)
{
  // Test case where factor-based adjustment is used for slowest dimension
  constexpr auto X = 100;
  constexpr auto Y = 100;
  constexpr auto Z = 100;

  constexpr auto X_CHUNK = 50;
  constexpr auto Y_CHUNK = 10;  // Small chunks create many tiles: 100/10 = 10
  constexpr auto Z_CHUNK = 10;  // Small chunks create many tiles: 100/10 = 10

  constexpr auto DATASET = "/factor_based_adjustment";
  constexpr auto DIM     = 3;
  const auto file_path   = base_path / "factor_based_adjustment.h5";

  constexpr auto dims       = std::array<hsize_t, DIM>{X, Y, Z};
  constexpr auto chunk_dims = std::array<hsize_t, DIM>{X_CHUNK, Y_CHUNK, Z_CHUNK};

  create_hdf5_file_with_sequential_data(file_path, DATASET, dims, &chunk_dims);

  // Request only 5 tiles, tiles_in_other_dims = 10 * 10 = 100
  // Factor = 100 / 5 = 20 (should increase slowest dim tile size)
  const auto num_tiles  = 5;
  const auto tile_shape = get_tile_shape_for_file(file_path, DATASET, dims, num_tiles);

  // Should succeed using factor-based adjustment
  ASSERT_TRUE(tile_shape.has_value());

  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  const auto& ts = *tile_shape;
  ASSERT_EQ(ts.size(), DIM);

  // Factor = 100/5 = 20, so slowest tile size = 50 * 20 = 1000
  ASSERT_EQ(ts[0], X);
  ASSERT_EQ(ts[1], Y_CHUNK);
  ASSERT_EQ(ts[2], Z_CHUNK);
}

TEST_F(IOHDF5TilingUnit, ChunkedPartitionerFactorLessThanOne)
{
  // Test case where factor < 1 (more tiles requested than tiles in other dims)
  constexpr auto X = 100;
  constexpr auto Y = 50;
  constexpr auto Z = 50;

  constexpr auto X_CHUNK = 50;
  constexpr auto Y_CHUNK = 50;
  constexpr auto Z_CHUNK = 50;

  constexpr auto DATASET = "/factor_less_than_one";
  constexpr auto DIM     = 3;
  const auto file_path   = base_path / "factor_less_than_one.h5";

  constexpr auto dims       = std::array<hsize_t, DIM>{X, Y, Z};
  constexpr auto chunk_dims = std::array<hsize_t, DIM>{X_CHUNK, Y_CHUNK, Z_CHUNK};

  create_hdf5_file_with_sequential_data(file_path, DATASET, dims, &chunk_dims);

  // Request 10 tiles, tiles_in_other_dims = 1 * 1 = 1
  // Factor = 1 / 10 = 0.1 (should decrease slowest dim tile size)
  const auto num_tiles  = 10;
  const auto tile_shape = get_tile_shape_for_file(file_path, DATASET, dims, num_tiles);

  // Should succeed using factor-based adjustment
  ASSERT_TRUE(tile_shape.has_value());

  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  const auto& ts = *tile_shape;
  ASSERT_EQ(ts.size(), DIM);

  // - tiles_needed = ceil(100/50) * ceil(50/50) * ceil(50/50) = 2 * 1 * 1 = 2
  // - adjustment_factor = 2/10 = 0.2
  // - Adjusted slowest tile size = 50 * 0.2 = 10
  ASSERT_EQ(ts[0], 10);
  ASSERT_EQ(ts[1], Y_CHUNK);
  ASSERT_EQ(ts[2], Z_CHUNK);
}

}  // namespace test_io_hdf5_tiling
