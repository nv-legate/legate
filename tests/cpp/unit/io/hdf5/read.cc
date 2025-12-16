/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/io/hdf5/interface.h>

#include <H5Fpublic.h>
#include <H5Gpublic.h>
#include <H5Ppublic.h>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <filesystem>
#include <string_view>
#include <utilities/utilities.h>

namespace test_io_hdf5_read {

namespace {

/**
 * @brief Helper function to create HDF5 file with sequential float data
 *
 * @param file_path The path to the HDF5 file.
 * @param dataset_name The name of the dataset.
 * @param dims The dimensions of the dataset.
 */
template <std::size_t NDIM>
void create_hdf5_file_with_sequential_data(const std::filesystem::path& file_path,
                                           const std::string& dataset_name,
                                           const std::array<hsize_t, NDIM>& dims)
{
  const auto file = H5Fcreate(file_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  ASSERT_GE(file, 0);

  const auto space = H5Screate_simple(dims.size(), dims.data(), nullptr);
  ASSERT_GE(space, 0);

  const auto dset = H5Dcreate(
    file, dataset_name.c_str(), H5T_IEEE_F32LE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
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
  ASSERT_GE(H5Sclose(space), 0);
  ASSERT_GE(H5Dclose(dset), 0);
  ASSERT_GE(H5Fclose(file), 0);
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
  auto* runtime = legate::Runtime::get_runtime();
  auto lib      = runtime->find_library(Config::LIBRARY_NAME);

  auto verify_task = runtime->create_task(lib, Verify3DTask::TASK_CONFIG.task_id());
  verify_task.add_input(read_array);
  verify_task.add_scalar_arg(legate::Scalar{Y});
  verify_task.add_scalar_arg(legate::Scalar{Z});
  runtime->submit(std::move(verify_task));
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
  EXPECT_EQ(read_array.shape().volume(), X * Y);
  EXPECT_EQ(read_array.type(), legate::float32());
  EXPECT_EQ(read_array.dim(), DIM);

  // Verify the data pattern
  auto* runtime = legate::Runtime::get_runtime();
  auto lib      = runtime->find_library(Config::LIBRARY_NAME);

  auto verify_task = runtime->create_task(lib, Verify2DTask::TASK_CONFIG.task_id());
  verify_task.add_input(read_array);
  verify_task.add_scalar_arg(legate::Scalar{Y});
  runtime->submit(std::move(verify_task));
}

}  // namespace test_io_hdf5_read
