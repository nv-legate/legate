/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/experimental/io/detail/library.h>
#include <legate/io/hdf5/detail/combine_vds.h>
#include <legate/io/hdf5/interface.h>

#include <H5Dpublic.h>
#include <H5Fpublic.h>
#include <H5Ppublic.h>
#include <H5Spublic.h>
#include <H5Tpublic.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <utilities/utilities.h>
#include <vector>

namespace test_io_hdf5_negative {

namespace {

/**
 * @brief Helper function to create an HDF5 file with variable-length string data.
 */
void create_hdf5_file_with_strings(const std::filesystem::path& file_path,
                                   const std::string& dataset_name,
                                   const std::vector<std::string>& strings)
{
  const auto file = H5Fcreate(file_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  ASSERT_GE(file, 0);

  const auto dims  = std::array<hsize_t, 1>{strings.size()};
  const auto space = H5Screate_simple(dims.size(), dims.data(), nullptr);

  ASSERT_GE(space, 0);

  // Create variable-length string type
  const auto str_type = H5Tcopy(H5T_C_S1);

  ASSERT_GE(str_type, 0);
  ASSERT_GE(H5Tset_size(str_type, H5T_VARIABLE), 0);

  const auto dset =
    H5Dcreate(file, dataset_name.c_str(), str_type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  ASSERT_GE(dset, 0);

  // Convert strings to C-style string pointers
  auto c_strs = std::vector<const char*>{};

  c_strs.reserve(strings.size());
  for (const auto& s : strings) {
    c_strs.push_back(s.c_str());
  }

  ASSERT_GE(
    H5Dwrite(
      dset, str_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, static_cast<const void*>(c_strs.data())),
    0);
  ASSERT_GE(H5Tclose(str_type), 0);
  ASSERT_GE(H5Sclose(space), 0);
  ASSERT_GE(H5Dclose(dset), 0);
  ASSERT_GE(H5Fclose(file), 0);
}

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_io_hdf5_negative";

  static void registration_callback(legate::Library /*library*/) {}
};

class IOHDF5NegativeTest : public RegisterOnceFixture<Config> {
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
  static inline auto base_path = std::filesystem::temp_directory_path() /
                                 (std::string{"legate_"} + std::string{Config::LIBRARY_NAME});
};

using IOHDF5NegativeDeathTest = IOHDF5NegativeTest;

}  // namespace

// Test that from_file throws when file does not exist
TEST_F(IOHDF5NegativeTest, FromFileFileNotFound)
{
  const auto non_existent_file = base_path / "non_existent_file.h5";

  // Ensure file does not exist
  static_cast<void>(std::filesystem::remove(non_existent_file));
  ASSERT_FALSE(std::filesystem::exists(non_existent_file));

  // Should throw std::system_error with no_such_file_or_directory
  ASSERT_THAT([&]() { std::ignore = legate::io::hdf5::from_file(non_existent_file, "dataset"); },
              testing::Throws<std::system_error>((::testing::Property(
                &std::system_error::code,
                ::testing::Eq(std::make_error_code(std::errc::no_such_file_or_directory))))));
}

// Test that from_file throws InvalidDataSetError when dataset does not exist
// This also covers InvalidDataSetError::path() and dataset_name()
TEST_F(IOHDF5NegativeTest, FromFileDatasetNotFound)
{
  const auto file_path    = base_path / "empty_file.h5";
  const auto dataset_name = std::string{"non_existent_dataset"};

  // Create an empty HDF5 file (no datasets)
  {
    const auto file = H5Fcreate(file_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    ASSERT_GE(file, 0);
    ASSERT_GE(H5Fclose(file), 0);
  }

  // Should throw InvalidDataSetError when dataset doesn't exist
  ASSERT_THROW(
    {
      try {
        std::ignore = legate::io::hdf5::from_file(file_path, dataset_name);
      } catch (const legate::io::hdf5::InvalidDataSetError& e) {
        // Verify the exception contains correct information
        // This covers InvalidDataSetError::path() and dataset_name()
        ASSERT_EQ(e.path().filename(), file_path.filename());
        ASSERT_EQ(e.dataset_name(), dataset_name);
        ASSERT_THAT(e.what(), ::testing::HasSubstr("does not exist"));
        throw;  // rethrow so ASSERT_THROW can verify the exception type
      }
    },
    legate::io::hdf5::InvalidDataSetError);
}

// Test that to_file throws when file path is a directory
TEST_F(IOHDF5NegativeTest, ToFileFilePathIsDirectory)
{
  auto* const runtime = legate::Runtime::get_runtime();

  // Create a simple array to write
  const auto shape    = legate::Shape{5};
  const auto array    = runtime->create_array(shape, legate::int32());
  const auto dir_path = base_path / "test_directory";

  std::filesystem::create_directories(dir_path);
  ASSERT_TRUE(std::filesystem::is_directory(dir_path));

  // Should throw std::invalid_argument when path is a directory
  ASSERT_THAT([&]() { legate::io::hdf5::to_file(array, dir_path, "dataset"); },
              testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("must be the name of a file, not a directory")));
}

// Test that reading string type aborts (not supported)
TEST_F(IOHDF5NegativeDeathTest, FromFileStringTypeNotSupported)
{
  constexpr auto DATASET = "string_dataset";
  const auto file_path   = base_path / "string_death_test.h5";
  const auto strings     = std::vector<std::string>{"hello", "world", "test"};

  // Note: File creation must be inside ASSERT_DEATH block because death tests fork the process.
  // If created outside, the forked child process cannot lock the already-locked HDF5 file.
  ASSERT_DEATH(
    {
      create_hdf5_file_with_strings(file_path, DATASET, strings);

      const auto read_array = legate::io::hdf5::from_file(file_path, DATASET);
      // Verify type deduction works correctly
      if (read_array.type() != legate::string_type()) {
        std::exit(1);  // Wrong type - fail differently
      }
      // This will trigger the read task and abort
      auto* runtime = legate::Runtime::get_runtime();
      runtime->issue_execution_fence(/* block */ true);
    },
    "(HDF5Read.*threw an unexpected exception|Data store of a nested array cannot be retrieved)");
}

}  // namespace test_io_hdf5_negative
