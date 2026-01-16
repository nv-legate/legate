/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/io/hdf5/interface.h>

#include <H5Fpublic.h>
#include <H5Gpublic.h>
#include <H5Ppublic.h>
#include <H5Tpublic.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <utilities/utilities.h>
#include <vector>

namespace test_io_hdf5_read {

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
 * @param num_files Number of files along the slowest dimension.
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

        src_data[local_index] = static_cast<float>(global_index);
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
    src_dims[0]                        = this_tile_dim;

    std::array<hsize_t, NDIM> vds_start{};
    vds_start[0] = start_idx;
    std::array<hsize_t, NDIM> vds_count{};
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

struct OpaqueType {
  std::uint8_t byte{};

  bool operator==(const OpaqueType& other) const { return byte == other.byte; }
};

constexpr auto MAGIC_BYTE = OpaqueType{123};

class Verify2DTask : public legate::LegateTask<Verify2DTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{2}}.with_signature(
      legate::TaskSignature{}.inputs(1).scalars(1));

  static void cpu_variant(legate::TaskContext ctx)
  {
    const auto& array       = ctx.input(0);
    const auto global_y_dim = ctx.scalar(0).value<std::uint32_t>();  // Global Y dimension
    const auto shape        = array.shape<2>();
    const auto acc          = array.data().read_accessor<float, 2>();

    for (auto it = legate::PointInRectIterator<2>{shape}; it.valid(); ++it) {
      const auto point = *it;

      // we should be tiling only on the slow dimension
      const auto global_index = (point[0] * global_y_dim) + point[1];

      const auto expected = static_cast<float>(global_index);
      const auto actual   = acc[point];

      EXPECT_EQ(actual, expected);
    }
  }
};

class Verify3DTask : public legate::LegateTask<Verify3DTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}}.with_signature(
      legate::TaskSignature{}.inputs(1).scalars(2));

  static void cpu_variant(legate::TaskContext ctx)
  {
    const auto array        = ctx.input(0);
    const auto global_y_dim = ctx.scalar(0).value<std::uint32_t>();  // Global Y dimension
    const auto global_z_dim = ctx.scalar(1).value<std::uint32_t>();  // Global Z dimension
    const auto shape        = array.shape<3>();
    const auto acc          = array.data().read_accessor<float, 3>();

    for (auto it = legate::PointInRectIterator<3>{shape}; it.valid(); ++it) {
      const auto point = *it;
      const auto global_index =
        (point[0] * global_y_dim * global_z_dim) + (point[1] * global_z_dim) + point[2];

      const auto expected = static_cast<float>(global_index);
      const auto actual   = acc[point];

      EXPECT_EQ(actual, expected);
    }
  }
};

class CheckerTask : public legate::LegateTask<CheckerTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}}.with_signature(legate::TaskSignature{}.inputs(1));

  static void cpu_variant(legate::TaskContext ctx)
  {
    constexpr auto DIM = 1;

    const auto array = ctx.input(0);
    const auto shape = array.shape<DIM>();
    const auto acc   = array.data().read_accessor<OpaqueType, DIM, /* VALIDATE_TYPE */ false>();

    for (auto it = legate::PointInRectIterator<DIM>{shape}; it.valid(); ++it) {
      ASSERT_EQ(acc[*it], MAGIC_BYTE);
    }
  }
};

/**
 * @brief Test parameter for parameterized HDF5 type deduction tests.
 */
struct TypeTestParam {
  hid_t hdf5_type;           ///< HDF5 native type
  legate::Type legate_type;  ///< Expected legate type
  std::string name;          ///< Test name for display
};

/**
 * @brief Helper function to create an HDF5 file with data of a specific type.
 *
 * @tparam T The C++ type of the data.
 * @param file_path The path to the HDF5 file.
 * @param dataset_name The name of the dataset.
 * @param hdf5_type The HDF5 type identifier.
 * @param data The data to write.
 */
template <typename T>
void create_hdf5_file_with_type(const std::filesystem::path& file_path,
                                const std::string& dataset_name,
                                hid_t hdf5_type,
                                const std::vector<T>& data)
{
  const auto file = H5Fcreate(file_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  ASSERT_GE(file, 0);

  const auto dims  = std::array<hsize_t, 1>{data.size()};
  const auto space = H5Screate_simple(dims.size(), dims.data(), nullptr);
  ASSERT_GE(space, 0);

  const auto dset =
    H5Dcreate(file, dataset_name.c_str(), hdf5_type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  ASSERT_GE(dset, 0);

  ASSERT_GE(H5Dwrite(dset, hdf5_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()), 0);
  ASSERT_GE(H5Sclose(space), 0);
  ASSERT_GE(H5Dclose(dset), 0);
  ASSERT_GE(H5Fclose(file), 0);
}

/**
 * @brief Helper function to create an HDF5 file with boolean data.
 *
 * This uses an enum type with TRUE/FALSE members, which is how legate stores boolean data
 * in HDF5 files. H5T_NATIVE_HBOOL is just an alias for std::uint8_t on most systems and would
 * be read back as UNSIGNED_INTEGER, not BOOL.
 *
 * @param file_path The path to the HDF5 file.
 * @param dataset_name The name of the dataset.
 * @param size The number of elements.
 */
void create_hdf5_file_with_bool(const std::filesystem::path& file_path,
                                const std::string& dataset_name,
                                std::size_t size)
{
  const auto file = H5Fcreate(file_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  ASSERT_GE(file, 0);

  const auto dims  = std::array<hsize_t, 1>{size};
  const auto space = H5Screate_simple(dims.size(), dims.data(), nullptr);

  ASSERT_GE(space, 0);

  // Create a boolean enum type with TRUE/FALSE members (same as legate's writer)
  // H5T_NATIVE_HBOOL is just std::uint8_t and won't be recognized as boolean when reading
  const auto bool_type = H5Tenum_create(H5T_NATIVE_INT8);

  ASSERT_GE(bool_type, 0);

  constexpr std::int8_t false_val = 0;
  constexpr std::int8_t true_val  = 1;

  ASSERT_GE(H5Tenum_insert(bool_type, "FALSE", &false_val), 0);
  ASSERT_GE(H5Tenum_insert(bool_type, "TRUE", &true_val), 0);

  const auto dset =
    H5Dcreate(file, dataset_name.c_str(), bool_type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  ASSERT_GE(dset, 0);

  // Use a raw array instead of std::vector<bool> which is a problematic specialization
  auto data = std::make_unique<std::int8_t[]>(size);

  for (std::size_t i = 0; i < size; ++i) {
    data[i] = (i % 2) == 0 ? 1 : 0;
  }

  ASSERT_GE(H5Dwrite(dset, bool_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.get()), 0);
  ASSERT_GE(H5Tclose(bool_type), 0);
  ASSERT_GE(H5Sclose(space), 0);
  ASSERT_GE(H5Dclose(dset), 0);
  ASSERT_GE(H5Fclose(file), 0);
}

/**
 * @brief Helper to create HDF5 file with sequential data of type T.
 */
template <typename T>
void create_hdf5_file_with_sequential_type(const std::filesystem::path& file_path,
                                           const std::string& dataset_name,
                                           hid_t hdf5_type,
                                           std::size_t size)
{
  auto data = std::vector<T>(size);

  for (std::size_t i = 0; i < size; ++i) {
    data[i] = static_cast<T>(i);
  }
  create_hdf5_file_with_type(file_path, dataset_name, hdf5_type, data);
}

/**
 * @brief Get the HDF5 type for float16 (IEEE 754 half-precision).
 */
[[nodiscard]] hid_t get_float16_hdf5_type()
{
  // IEEE 754 half-precision (float16) bit layout:
  // - Sign bit at position 15
  // - Exponent: 5 bits starting at position 10
  // - Mantissa: 10 bits starting at position 0
  // - Exponent bias: 15
  constexpr auto SIGN_BIT_POS  = 15;
  constexpr auto EXP_BIT_POS   = 10;
  constexpr auto EXP_BITS      = 5;
  constexpr auto MANTISSA_POS  = 0;
  constexpr auto MANTISSA_BITS = 10;
  constexpr auto EXP_BIAS      = 15;
  constexpr auto FLOAT16_SIZE  = 2;

  const auto type = H5Tcopy(H5T_IEEE_F32LE);

  H5Tset_fields(type, SIGN_BIT_POS, EXP_BIT_POS, EXP_BITS, MANTISSA_POS, MANTISSA_BITS);
  H5Tset_size(type, FLOAT16_SIZE);
  H5Tset_ebias(type, EXP_BIAS);
  return type;
}

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_io_hdf5_read";

  static void registration_callback(legate::Library library)
  {
    CheckerTask::register_variants(library);
    Verify2DTask::register_variants(library);
    Verify3DTask::register_variants(library);
  }
};

class IOHDF5ReadUnit : public RegisterOnceFixture<Config> {
 public:
  void SetUp() override
  {
    RegisterOnceFixture::SetUp();
    ASSERT_NO_THROW(std::filesystem::create_directories(base_path));
  }

  void TearDown() override
  {
    RegisterOnceFixture::TearDown();
    ASSERT_NO_THROW(static_cast<void>(std::filesystem::remove_all(base_path)));
  }

  // NOLINTNEXTLINE(cert-err58-cpp)
  static inline auto base_path = std::filesystem::temp_directory_path() / "legate";
};

class IOHDF5ReadTypeDeduction : public RegisterOnceFixture<Config>,
                                public ::testing::WithParamInterface<TypeTestParam> {
 public:
  void SetUp() override
  {
    RegisterOnceFixture::SetUp();
    ASSERT_NO_THROW(std::filesystem::create_directories(base_path));
  }

  void TearDown() override
  {
    RegisterOnceFixture::TearDown();
    ASSERT_NO_THROW(static_cast<void>(std::filesystem::remove_all(base_path)));
  }

  // NOLINTNEXTLINE(cert-err58-cpp)
  static inline auto base_path = std::filesystem::temp_directory_path() / "legate_type_tests";
};

// NOLINTNEXTLINE(cert-err58-cpp)
INSTANTIATE_TEST_SUITE_P(
  IOHDF5ReadUnit,
  IOHDF5ReadTypeDeduction,
  ::testing::Values(TypeTestParam{H5T_NATIVE_HBOOL, legate::bool_(), "bool"},
                    TypeTestParam{H5T_NATIVE_INT8, legate::int8(), "int8"},
                    TypeTestParam{H5T_NATIVE_INT16, legate::int16(), "int16"},
                    TypeTestParam{H5T_NATIVE_INT32, legate::int32(), "int32"},
                    TypeTestParam{H5T_NATIVE_INT64, legate::int64(), "int64"},
                    TypeTestParam{H5T_NATIVE_UINT8, legate::uint8(), "uint8"},
                    TypeTestParam{H5T_NATIVE_UINT16, legate::uint16(), "uint16"},
                    TypeTestParam{H5T_NATIVE_UINT32, legate::uint32(), "uint32"},
                    TypeTestParam{H5T_NATIVE_UINT64, legate::uint64(), "uint64"},
                    TypeTestParam{get_float16_hdf5_type(), legate::float16(), "float16"},
                    TypeTestParam{H5T_NATIVE_FLOAT, legate::float32(), "float32"},
                    TypeTestParam{H5T_NATIVE_DOUBLE, legate::float64(), "float64"}),
  [](const ::testing::TestParamInfo<TypeTestParam>& param_info) { return param_info.param.name; });

void submit_verify_2d_task(const legate::LogicalArray& read_array, std::uint32_t y)
{
  auto* runtime    = legate::Runtime::get_runtime();
  auto lib         = runtime->find_library(Config::LIBRARY_NAME);
  auto verify_task = runtime->create_task(lib, Verify2DTask::TASK_CONFIG.task_id());

  verify_task.add_input(read_array);
  verify_task.add_scalar_arg(legate::Scalar{y});
  runtime->submit(std::move(verify_task));
}

void submit_verify_3d_task(const legate::LogicalArray& read_array, std::uint32_t y, std::uint32_t z)
{
  auto* runtime    = legate::Runtime::get_runtime();
  auto lib         = runtime->find_library(Config::LIBRARY_NAME);
  auto verify_task = runtime->create_task(lib, Verify3DTask::TASK_CONFIG.task_id());

  verify_task.add_input(read_array);
  verify_task.add_scalar_arg(legate::Scalar{y});
  verify_task.add_scalar_arg(legate::Scalar{z});
  runtime->submit(std::move(verify_task));
}

}  // namespace

TEST_F(IOHDF5ReadUnit, Binary)
{
  constexpr auto SIZE         = 10;
  constexpr auto BASE_DATASET = "/legate";
  constexpr auto DATASET      = "/legate/foo";
  constexpr auto DIM          = 1;
  const auto file_path        = base_path / "foo.hdf";

  {
    /*
     * Create the source files and  datasets. Write data to each dataset and
     * close all resources.
     */

    const auto opaque_hid = H5Tcreate(H5T_OPAQUE, sizeof(OpaqueType));

    ASSERT_GE(opaque_hid, 0);

    const auto file = H5Fcreate(file_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    ASSERT_GE(file, 0);

    constexpr auto dims = std::array<hsize_t, 1>{SIZE};
    const auto space    = H5Screate_simple(dims.size(), dims.data(), nullptr);

    ASSERT_GE(space, 0);

    const auto sub_group = H5Gcreate(file, BASE_DATASET, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    ASSERT_GE(sub_group, 0);

    const auto dset =
      H5Dcreate(file, DATASET, opaque_hid, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    ASSERT_GE(dset, 0);

    auto data = std::array<OpaqueType, SIZE>{};

    data.fill(MAGIC_BYTE);

    ASSERT_GE(H5Dwrite(dset, opaque_hid, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()), 0);
    ASSERT_GE(H5Sclose(space), 0);
    ASSERT_GE(H5Dclose(dset), 0);
    ASSERT_GE(H5Fclose(file), 0);
    ASSERT_GE(H5Tclose(opaque_hid), 0);
    ASSERT_GE(H5Gclose(sub_group), 0);
  }

  const auto read_array = legate::io::hdf5::from_file(file_path, DATASET);

  ASSERT_EQ(read_array.shape(), legate::Shape{SIZE});
  ASSERT_EQ(read_array.type(), legate::binary_type(1));
  ASSERT_EQ(read_array.dim(), DIM);

  auto* runtime     = legate::Runtime::get_runtime();
  auto lib          = runtime->find_library(Config::LIBRARY_NAME);
  auto checker_task = runtime->create_task(lib, CheckerTask::TASK_CONFIG.task_id());

  checker_task.add_input(read_array);
  runtime->submit(std::move(checker_task));
}

TEST_F(IOHDF5ReadUnit, ThreeDimensional)
{
  constexpr auto X       = 400;
  constexpr auto Y       = 300;
  constexpr auto Z       = 200;
  constexpr auto DATASET = "/three_dimensional";
  constexpr auto DIM     = 3;
  const auto file_path   = base_path / "three_d.h5";

  // Create 3D HDF5 file with sequential data
  constexpr auto dims = std::array<hsize_t, 3>{X, Y, Z};

  create_hdf5_file_with_sequential_data(file_path, DATASET, dims);

  // Read the 3D array back using Legate HDF5 interface
  const auto read_array = legate::io::hdf5::from_file(file_path, DATASET);

  // Verify dimensions and shape
  ASSERT_EQ(read_array.shape().volume(), X * Y * Z);
  ASSERT_EQ(read_array.type(), legate::float32());
  ASSERT_EQ(read_array.dim(), DIM);

  // Verify the data pattern
  submit_verify_3d_task(read_array, Y, Z);
}

TEST_F(IOHDF5ReadUnit, TwoDimensional)
{
  constexpr auto X       = 100;
  constexpr auto Y       = 500;
  constexpr auto DATASET = "/two_dimensional";
  constexpr auto DIM     = 2;
  const auto file_path   = base_path / "two_d.h5";

  // Create 2D HDF5 file with sequential data
  constexpr auto dims = std::array<hsize_t, 2>{X, Y};

  create_hdf5_file_with_sequential_data(file_path, DATASET, dims);

  // Read the 2D array back using Legate HDF5 interface
  const auto read_array = legate::io::hdf5::from_file(file_path, DATASET);

  // Verify dimensions and shape
  ASSERT_EQ(read_array.shape().volume(), X * Y);
  ASSERT_EQ(read_array.type(), legate::float32());
  ASSERT_EQ(read_array.dim(), DIM);

  // Verify the data pattern
  submit_verify_2d_task(read_array, Y);
}

TEST_P(IOHDF5ReadTypeDeduction, DeduceType)
{
  constexpr auto SIZE    = 10;
  constexpr auto DATASET = "/test_dataset";
  const auto& param      = GetParam();
  const auto file_path   = base_path / (param.name + ".h5");

  // Create HDF5 file with appropriate data based on type
  if (param.legate_type == legate::bool_()) {
    create_hdf5_file_with_bool(file_path, DATASET, SIZE);
  } else if (param.legate_type == legate::int8()) {
    create_hdf5_file_with_sequential_type<std::int8_t>(file_path, DATASET, param.hdf5_type, SIZE);
  } else if (param.legate_type == legate::int16()) {
    create_hdf5_file_with_sequential_type<std::int16_t>(file_path, DATASET, param.hdf5_type, SIZE);
  } else if (param.legate_type == legate::int32()) {
    create_hdf5_file_with_sequential_type<std::int32_t>(file_path, DATASET, param.hdf5_type, SIZE);
  } else if (param.legate_type == legate::int64()) {
    create_hdf5_file_with_sequential_type<std::int64_t>(file_path, DATASET, param.hdf5_type, SIZE);
  } else if (param.legate_type == legate::uint8()) {
    create_hdf5_file_with_sequential_type<std::uint8_t>(file_path, DATASET, param.hdf5_type, SIZE);
  } else if (param.legate_type == legate::uint16() || param.legate_type == legate::float16()) {
    // float16 stored as uint16 bit pattern
    create_hdf5_file_with_sequential_type<std::uint16_t>(file_path, DATASET, param.hdf5_type, SIZE);
  } else if (param.legate_type == legate::uint32()) {
    create_hdf5_file_with_sequential_type<std::uint32_t>(file_path, DATASET, param.hdf5_type, SIZE);
  } else if (param.legate_type == legate::uint64()) {
    create_hdf5_file_with_sequential_type<std::uint64_t>(file_path, DATASET, param.hdf5_type, SIZE);
  } else if (param.legate_type == legate::float32()) {
    create_hdf5_file_with_sequential_type<float>(file_path, DATASET, param.hdf5_type, SIZE);
  } else if (param.legate_type == legate::float64()) {
    create_hdf5_file_with_sequential_type<double>(file_path, DATASET, param.hdf5_type, SIZE);
  } else {
    GTEST_FAIL() << "Unhandled type in test: " << param.name;
  }

  // Read the array back and verify type deduction
  const auto read_array = legate::io::hdf5::from_file(file_path, DATASET);

  ASSERT_EQ(read_array.shape(), legate::Shape{SIZE});
  ASSERT_EQ(read_array.type(), param.legate_type)
    << "Type mismatch for " << param.name << ": expected " << param.legate_type.to_string()
    << ", got " << read_array.type().to_string();
  ASSERT_EQ(read_array.dim(), 1);
}

TEST_F(IOHDF5ReadUnit, ThreeDimensionalChunked)
{
  constexpr auto X       = 400;
  constexpr auto Y       = 300;
  constexpr auto Z       = 200;
  constexpr auto DATASET = "/three_dimensional";
  constexpr auto DIM     = 3;
  const auto file_path   = base_path / "three_d.h5";

  // Create 3D HDF5 file with sequential data
  constexpr auto dims       = std::array<hsize_t, 3>{X, Y, Z};
  constexpr auto chunk_dims = std::array<hsize_t, 3>{50, 300, 200};

  create_hdf5_file_with_sequential_data(file_path, DATASET, dims, &chunk_dims);

  // Read the 3D array back using Legate HDF5 interface
  const auto read_array = legate::io::hdf5::from_file(file_path, DATASET);

  // Verify dimensions and shape
  ASSERT_EQ(read_array.shape().volume(), X * Y * Z);
  ASSERT_EQ(read_array.type(), legate::float32());
  ASSERT_EQ(read_array.dim(), DIM);

  // Verify the data pattern
  submit_verify_3d_task(read_array, Y, Z);
}

TEST_F(IOHDF5ReadUnit, TwoDimensionalChunked)
{
  constexpr auto X       = 100;
  constexpr auto Y       = 500;
  constexpr auto DATASET = "/two_dimensional";
  constexpr auto DIM     = 2;
  const auto file_path   = base_path / "two_d.h5";

  // Create 2D HDF5 file with sequential data
  constexpr auto dims       = std::array<hsize_t, 2>{X, Y};
  constexpr auto chunk_dims = std::array<hsize_t, 2>{50, 100};

  create_hdf5_file_with_sequential_data(file_path, DATASET, dims, &chunk_dims);

  // Read the 2D array back using Legate HDF5 interface
  const auto read_array = legate::io::hdf5::from_file(file_path, DATASET);

  // Verify dimensions and shape
  ASSERT_EQ(read_array.shape().volume(), X * Y);
  ASSERT_EQ(read_array.type(), legate::float32());
  ASSERT_EQ(read_array.dim(), DIM);

  // Verify the data pattern
  submit_verify_2d_task(read_array, Y);
}

TEST_F(IOHDF5ReadUnit, TwoDimensionalVDS)
{
  constexpr auto X         = 100;
  constexpr auto Y         = 500;
  constexpr auto NUM_FILES = 4;
  constexpr auto DATASET   = "/two_dimensional";
  constexpr auto DIM       = 2;
  const auto vds_file_path = base_path / "two_d_vds.h5";
  const auto source_dir    = base_path / "two_d_vds_source";

  // Create 2D HDF5 file with sequential data
  constexpr auto dims = std::array<hsize_t, 2>{X, Y};

  create_vds_with_sequential_data(vds_file_path, source_dir, DATASET, dims, NUM_FILES);

  // Read the 2D array back using Legate HDF5 interface
  const auto read_array = legate::io::hdf5::from_file(vds_file_path, DATASET);

  // Verify dimensions and shape
  ASSERT_EQ(read_array.shape().volume(), X * Y);
  ASSERT_EQ(read_array.type(), legate::float32());
  ASSERT_EQ(read_array.dim(), DIM);

  // Verify the data pattern
  submit_verify_2d_task(read_array, Y);
}

TEST_F(IOHDF5ReadUnit, ThreeDimensionalVDS)
{
  constexpr auto X         = 100;
  constexpr auto Y         = 500;
  constexpr auto Z         = 200;
  constexpr auto NUM_FILES = 4;
  constexpr auto DATASET   = "/three_dimensional";
  constexpr auto DIM       = 3;
  const auto vds_file_path = base_path / "three_d_vds.h5";
  const auto source_dir    = base_path / "three_d_vds_source";

  // Create 2D HDF5 file with sequential data
  constexpr auto dims = std::array<hsize_t, DIM>{X, Y, Z};

  create_vds_with_sequential_data(vds_file_path, source_dir, DATASET, dims, NUM_FILES);

  // Read the 2D array back using Legate HDF5 interface
  const auto read_array = legate::io::hdf5::from_file(vds_file_path, DATASET);

  // Verify dimensions and shape
  ASSERT_EQ(read_array.shape().volume(), X * Y * Z);
  ASSERT_EQ(read_array.type(), legate::float32());
  ASSERT_EQ(read_array.dim(), DIM);

  // Verify the data pattern
  submit_verify_3d_task(read_array, Y, Z);
}

TEST_F(IOHDF5ReadUnit, TwoDimensionalVDSChunked)
{
  constexpr auto X         = 100;
  constexpr auto Y         = 500;
  constexpr auto NUM_FILES = 4;
  constexpr auto DATASET   = "/two_dimensional";
  constexpr auto DIM       = 2;
  const auto vds_file_path = base_path / "two_d_vds.h5";
  const auto source_dir    = base_path / "two_d_vds_source";

  // Create 2D HDF5 file with chunked data
  constexpr auto dims       = std::array<hsize_t, 2>{X, Y};
  constexpr auto chunk_dims = std::array<hsize_t, 2>{50, 100};

  create_vds_with_sequential_data(vds_file_path, source_dir, DATASET, dims, NUM_FILES, &chunk_dims);

  // Read the 2D array back using Legate HDF5 interface
  const auto read_array = legate::io::hdf5::from_file(vds_file_path, DATASET);

  // Verify dimensions and shape
  ASSERT_EQ(read_array.shape().volume(), X * Y);
  ASSERT_EQ(read_array.type(), legate::float32());
  ASSERT_EQ(read_array.dim(), DIM);

  // Verify the data pattern
  submit_verify_2d_task(read_array, Y);
}

}  // namespace test_io_hdf5_read
