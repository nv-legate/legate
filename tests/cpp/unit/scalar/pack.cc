/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/scalar.h>
#include <legate/utilities/detail/buffer_builder.h>
#include <legate/utilities/detail/deserializer.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace pack_scalar_test {

namespace {

using PackScalarUnit = DefaultFixture;

constexpr bool BOOL_VALUE            = true;
constexpr std::int32_t INT32_VALUE   = 2700;
constexpr std::uint32_t UINT32_VALUE = 999;
constexpr std::uint64_t UINT64_VALUE = 100;
constexpr std::uint32_t DATA_SIZE    = 10;

struct PaddingStructData {
  bool bool_data;
  std::int32_t int32_data;
  std::uint64_t uint64_data;

  bool operator==(const PaddingStructData& other) const
  {
    return bool_data == other.bool_data && int32_data == other.int32_data &&
           uint64_data == other.uint64_data;
  }
};

struct [[gnu::packed]] NoPaddingStructData {
  bool bool_data;
  std::int32_t int32_data;
  std::uint64_t uint64_data;

  bool operator==(const NoPaddingStructData& other) const
  {
    return bool_data == other.bool_data && int32_data == other.int32_data &&
           uint64_data == other.uint64_data;
  }
};

static_assert(sizeof(PaddingStructData) > sizeof(NoPaddingStructData));

class ScalarUnitTestDeserializer
  : public legate::detail::BaseDeserializer<ScalarUnitTestDeserializer> {
 public:
  ScalarUnitTestDeserializer(const void* args, std::size_t arglen);

  using BaseDeserializer::unpack_impl;
};

ScalarUnitTestDeserializer::ScalarUnitTestDeserializer(const void* args, std::size_t arglen)
  : BaseDeserializer{args, arglen}
{
}

class ScalarDimTest : public PackScalarUnit, public ::testing::WithParamInterface<std::int32_t> {};

INSTANTIATE_TEST_SUITE_P(PackScalarUnit, ScalarDimTest, ::testing::Range(1, LEGATE_MAX_DIM));

void check_pack(const legate::Scalar& scalar)
{
  legate::detail::BufferBuilder buf;
  scalar.impl()->pack(buf);
  auto legion_buffer = buf.to_legion_buffer();

  ASSERT_NE(legion_buffer.get_ptr(), nullptr);

  ScalarUnitTestDeserializer deserializer{legion_buffer.get_ptr(), legion_buffer.get_size()};
  auto scalar_unpack = deserializer.unpack_scalar();

  ASSERT_EQ(scalar_unpack->type()->code, scalar.type().code());
  ASSERT_EQ(scalar_unpack->size(), scalar.size());
}

class PackPointScalarFn {
 public:
  template <std::int32_t DIM>
  void operator()() const
  {
    auto point = legate::Point<DIM>::ONES();
    const legate::Scalar scalar{point};

    check_pack(scalar);
  }
};

class PackRectScalarFn {
 public:
  template <std::int32_t DIM>
  void operator()() const
  {
    auto rect = legate::Rect<DIM>{legate::Point<DIM>::ZEROES(), legate::Point<DIM>::ONES()};
    const legate::Scalar scalar{rect};

    check_pack(scalar);
  }
};

}  // namespace

TEST_F(PackScalarUnit, PackNullScalar) { check_pack(legate::Scalar{}); }

TEST_F(PackScalarUnit, PackFixedArrayScalar)
{
  const legate::Scalar scalar{std::vector<std::uint32_t>{UINT32_VALUE, UINT32_VALUE}};

  check_pack(scalar);
}

TEST_F(PackScalarUnit, PackSingleValueScalar)
{
  const legate::Scalar scalar{BOOL_VALUE};

  check_pack(scalar);
}

TEST_F(PackScalarUnit, PackStringScalar)
{
  const legate::Scalar scalar{"123"};

  check_pack(scalar);
}

TEST_F(PackScalarUnit, PackPaddingStructScalar)
{
  const PaddingStructData struct_data = {BOOL_VALUE, INT32_VALUE, UINT64_VALUE};
  const legate::Scalar scalar{
    struct_data,
    legate::struct_type(/* align */ true, legate::bool_(), legate::int32(), legate::uint64())};

  check_pack(scalar);
}

TEST_F(PackScalarUnit, PackNoPaddingStructScalar)
{
  const NoPaddingStructData struct_data = {BOOL_VALUE, INT32_VALUE, UINT64_VALUE};
  const legate::Scalar scalar{
    struct_data,
    legate::struct_type(/* align */ false, legate::bool_(), legate::int32(), legate::uint64())};

  check_pack(scalar);
}

TEST_F(PackScalarUnit, PackSharedScalar)
{
  const auto data_vec = std::vector<std::uint64_t>(DATA_SIZE, UINT64_VALUE);
  const auto* data    = data_vec.data();
  const legate::Scalar scalar{legate::uint64(), data, /* copy */ false};

  check_pack(scalar);
}

TEST_F(PackScalarUnit, PackOwnedSharedScalar)
{
  const auto data_vec = std::vector<std::uint64_t>(DATA_SIZE, UINT64_VALUE);
  const auto* data    = data_vec.data();
  const legate::Scalar scalar{legate::uint64(), data, /* copy */ true};

  check_pack(scalar);
}

TEST_F(PackScalarUnit, PackCopiedScalar)
{
  const legate::Scalar scalar1{INT32_VALUE};
  const legate::Scalar scalar2{scalar1};  // NOLINT(performance-unnecessary-copy-initialization)

  check_pack(scalar2);
}

TEST_P(ScalarDimTest, PackPointScalar)
{
  const auto DIM = GetParam();

  legate::dim_dispatch(DIM, PackPointScalarFn{});
}

TEST_P(ScalarDimTest, PackRectScalar)
{
  const auto DIM = GetParam();

  legate::dim_dispatch(DIM, PackRectScalarFn{});
}

}  // namespace pack_scalar_test
