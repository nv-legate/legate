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

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <filesystem>
#include <string_view>
#include <utilities/utilities.h>

namespace test_io_hdf5_read {

namespace {

struct OpaqueType {
  std::uint8_t byte{};

  bool operator==(const OpaqueType& other) const { return byte == other.byte; }
};

constexpr auto MAGIC_BYTE = OpaqueType{123};

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

}  // namespace test_io_hdf5_read
