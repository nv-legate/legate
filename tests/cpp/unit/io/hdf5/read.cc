/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HIGHFIVE_CXX_STD
#include <legate/utilities/cpp_version.h>
// Highfive doesn't export this in their cmake files, so we have to
#define HIGHFIVE_CXX_STD LEGATE_CPP_MIN_VERSION
#endif

#include <legate.h>

#include <legate/io/hdf5/interface.h>

#include <highfive/H5DataType.hpp>
#include <highfive/H5File.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <filesystem>
#include <string_view>
#include <utilities/utilities.h>

namespace {

struct OpaqueType {
  std::uint8_t byte{};

  bool operator==(const OpaqueType& other) const { return byte == other.byte; }
};

}  // namespace

namespace HighFive {

template <>
inline AtomicType<OpaqueType>::AtomicType()
{
  _hid = detail::h5t_create(H5T_OPAQUE, sizeof(OpaqueType));
}

}  // namespace HighFive

namespace test_io_hdf5_read {

namespace {

constexpr auto MAGIC_BYTE = OpaqueType{123};

class CheckerTask : public legate::LegateTask<CheckerTask> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{0};

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

// TODO(jfaibussowit)
// Re-enable this test once Quincey and I figure out why reading Opaque data as std::uint8_t
// magically works for the SLAC dataset
TEST_F(IOHDF5ReadUnit, Binary)
{
  constexpr auto SIZE    = 10;
  constexpr auto DATASET = "legate/foo";
  constexpr auto DIM     = 1;
  const auto file_path   = base_path / "foo.hdf";

  {
    auto hf_file = HighFive::File{file_path, HighFive::File::Truncate};
    auto dset    = hf_file.createDataSet<OpaqueType>(DATASET, HighFive::DataSpace{SIZE});

    // This is the whole point of this exercise, if this isn't true, then the below does not
    // exercise the binary-type load path
    ASSERT_EQ(dset.getDataType().getClass(), HighFive::DataTypeClass::Opaque);
    ASSERT_EQ(dset.getDataType().getSize(), sizeof(OpaqueType));

    auto data = std::array<OpaqueType, SIZE>{};

    data.fill(MAGIC_BYTE);
    dset.write(data);
  }

  const auto read_array = legate::io::hdf5::from_file(file_path, DATASET);

  ASSERT_EQ(read_array.shape(), legate::Shape{SIZE});
  ASSERT_EQ(read_array.type(), legate::binary_type(1));
  ASSERT_EQ(read_array.dim(), DIM);

  auto* runtime     = legate::Runtime::get_runtime();
  auto lib          = runtime->find_library(Config::LIBRARY_NAME);
  auto checker_task = runtime->create_task(lib, CheckerTask::TASK_ID);

  checker_task.add_input(read_array);
  runtime->submit(std::move(checker_task));
}

}  // namespace test_io_hdf5_read
