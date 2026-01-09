/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <stdexcept>
#include <utilities/utilities.h>

namespace logical_store_create_test {

namespace {

using LogicalStoreCreateUnit = DefaultFixture;

constexpr std::int32_t SCALAR_VALUE = 10;
constexpr float FLOAT_VALUE         = 10.0F;
constexpr double DOUBLE_VALUE       = 11.0;

class CreateStoreTest
  : public LogicalStoreCreateUnit,
    public ::testing::WithParamInterface<std::tuple<legate::Type, std::uint32_t>> {};

class CreateScalarStoreTest
  : public LogicalStoreCreateUnit,
    public ::testing::WithParamInterface<std::tuple<legate::Type, legate::Scalar>> {};

class CreateScalarStoreWithExtentsTest
  : public LogicalStoreCreateUnit,
    public ::testing::WithParamInterface<legate::tuple<std::uint64_t>> {};

class NegativeStoreDimTest : public LogicalStoreCreateUnit,
                             public ::testing::WithParamInterface<std::uint32_t> {};

class NegativeStoreTypeTest : public LogicalStoreCreateUnit,
                              public ::testing::WithParamInterface<legate::Type> {};

INSTANTIATE_TEST_SUITE_P(
  LogicalStoreCreateUnit,
  CreateStoreTest,
  ::testing::Combine(
    ::testing::Values(
      legate::bool_(),
      legate::int8(),
      legate::int16(),
      legate::int32(),
      legate::int64(),
      legate::uint8(),
      legate::uint16(),
      legate::uint32(),
      legate::uint64(),
      legate::float16(),
      legate::float32(),
      legate::float64(),
      legate::complex64(),
      legate::complex128(),
      legate::null_type(),
      legate::binary_type(10),
      static_cast<legate::Type>(legate::fixed_array_type(legate::bool_(), 2)),
      static_cast<legate::Type>(legate::struct_type(false, legate::bool_(), legate::uint32())),
      static_cast<legate::Type>(
        legate::struct_type(true, legate::int16(), legate::bool_(), legate::uint32()))),
    ::testing::Range(1U, static_cast<std::uint32_t>(LEGATE_MAX_DIM))));

INSTANTIATE_TEST_SUITE_P(
  LogicalStoreCreateUnit,
  CreateScalarStoreTest,
  ::testing::Values(
    std::make_tuple(legate::bool_(), legate::Scalar{static_cast<bool>(0)}),
    std::make_tuple(legate::int8(), legate::Scalar{static_cast<std::int8_t>(1)}),
    std::make_tuple(legate::int16(), legate::Scalar{static_cast<std::int16_t>(2)}),
    std::make_tuple(legate::int32(), legate::Scalar{static_cast<std::int32_t>(3)}),
    std::make_tuple(legate::int64(), legate::Scalar{static_cast<std::int64_t>(4)}),
    std::make_tuple(legate::uint8(), legate::Scalar{static_cast<std::uint8_t>(5)}),
    std::make_tuple(legate::uint16(), legate::Scalar{static_cast<std::uint16_t>(6)}),
    std::make_tuple(legate::uint32(), legate::Scalar{static_cast<std::uint32_t>(7)}),
    std::make_tuple(legate::uint64(), legate::Scalar{static_cast<std::uint64_t>(8)}),
    std::make_tuple(legate::float16(), legate::Scalar{static_cast<legate::Half>(FLOAT_VALUE)}),
    std::make_tuple(legate::float32(), legate::Scalar{FLOAT_VALUE}),
    std::make_tuple(legate::float64(), legate::Scalar{DOUBLE_VALUE}),
    std::make_tuple(legate::complex64(),
                    legate::Scalar{legate::Complex<float>{FLOAT_VALUE, FLOAT_VALUE}}),
    std::make_tuple(legate::complex128(),
                    legate::Scalar{legate::Complex<double>{DOUBLE_VALUE, DOUBLE_VALUE}})));

INSTANTIATE_TEST_SUITE_P(LogicalStoreCreateUnit,
                         CreateScalarStoreWithExtentsTest,
                         ::testing::Values(legate::tuple<std::uint64_t>{1},
                                           legate::tuple<std::uint64_t>{1, 1},
                                           legate::tuple<std::uint64_t>{1, 1, 1}));

INSTANTIATE_TEST_SUITE_P(LogicalStoreCreateUnit,
                         NegativeStoreDimTest,
                         ::testing::Values(LEGATE_MAX_DIM + 1));

INSTANTIATE_TEST_SUITE_P(LogicalStoreCreateUnit,
                         NegativeStoreTypeTest,
                         ::testing::Values(legate::string_type(),
                                           legate::list_type(legate::int32())));

legate::Shape generate_shape(std::uint32_t dim)
{
  legate::tuple<std::uint64_t> extents{};

  for (auto i = 1U; i <= dim; ++i) {
    extents.data().push_back(static_cast<std::uint64_t>(i));
  }

  return legate::Shape{extents};
}

}  // namespace

TEST_P(CreateStoreTest, UnboundStore)
{
  const auto [type, dim] = GetParam();
  auto runtime           = legate::Runtime::get_runtime();
  auto store             = runtime->create_store(type, dim);

  ASSERT_TRUE(store.unbound());
  ASSERT_EQ(store.dim(), dim);
  ASSERT_THROW(static_cast<void>(store.extents()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(store.volume()), std::invalid_argument);
  ASSERT_EQ(store.type().code(), type.code());
  ASSERT_FALSE(store.transformed());
  ASSERT_FALSE(store.has_scalar_storage());
}

TEST_P(NegativeStoreDimTest, UnboundStore)
{
  auto runtime = legate::Runtime::get_runtime();

  ASSERT_THROW(static_cast<void>(runtime->create_store(legate::int64(), GetParam())),
               std::out_of_range);
}

TEST_P(NegativeStoreTypeTest, UnboundStore)
{
  auto runtime = legate::Runtime::get_runtime();

  // create with variable size type
  ASSERT_THROW(static_cast<void>(runtime->create_store(GetParam(), LEGATE_MAX_DIM)),
               std::invalid_argument);
}

TEST_P(CreateStoreTest, BoundStore)
{
  const auto [type, dim] = GetParam();
  auto runtime           = legate::Runtime::get_runtime();
  auto shape             = generate_shape(dim);
  auto store             = runtime->create_store(shape, type);

  ASSERT_FALSE(store.unbound());
  ASSERT_EQ(store.dim(), shape.ndim());
  ASSERT_EQ(store.extents(), shape.extents());
  ASSERT_EQ(store.volume(), store.extents().volume());
  ASSERT_EQ(store.type().code(), type.code());
  ASSERT_FALSE(store.transformed());
  ASSERT_FALSE(store.has_scalar_storage());
}

TEST_P(NegativeStoreDimTest, BoundStore)
{
  auto runtime = legate::Runtime::get_runtime();
  auto shape   = generate_shape(GetParam());

  ASSERT_THROW(static_cast<void>(runtime->create_store(shape, legate::int32())), std::out_of_range);
}

TEST_P(NegativeStoreTypeTest, BoundStore)
{
  auto runtime = legate::Runtime::get_runtime();
  auto shape   = generate_shape(LEGATE_MAX_DIM);

  // create with variable size type
  ASSERT_THROW(static_cast<void>(runtime->create_store(shape, GetParam())), std::invalid_argument);
}

TEST_F(LogicalStoreCreateUnit, OptimizeScalar)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store1  = runtime->create_store(legate::Shape{1}, legate::int64(), true);

  ASSERT_TRUE(store1.get_physical_store().is_future());
  ASSERT_TRUE(store1.has_scalar_storage());

  auto store2 = runtime->create_store(legate::Shape{1, 2}, legate::int64(), true);

  ASSERT_FALSE(store2.get_physical_store().is_future());
  ASSERT_FALSE(store2.has_scalar_storage());

  auto store3 = runtime->create_store(legate::Shape{1}, legate::int64(), false);

  ASSERT_FALSE(store3.get_physical_store().is_future());
  ASSERT_FALSE(store3.has_scalar_storage());
}

TEST_F(LogicalStoreCreateUnit, EmptyShapeCreation)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Shape{}, legate::int64());

  ASSERT_EQ(store.extents(), legate::tuple<std::uint64_t>{});
  ASSERT_EQ(store.dim(), 0);
}

TEST_P(CreateScalarStoreTest, Basic)
{
  const auto [type, scalar] = GetParam();
  auto runtime              = legate::Runtime::get_runtime();
  auto store                = runtime->create_store(scalar);

  ASSERT_FALSE(store.unbound());
  ASSERT_EQ(store.dim(), 1);
  ASSERT_EQ(store.extents(), legate::tuple<std::uint64_t>{1});
  ASSERT_EQ(store.volume(), 1);
  ASSERT_EQ(store.type().code(), type.code());
  ASSERT_FALSE(store.transformed());
  ASSERT_TRUE(store.has_scalar_storage());
}

TEST_P(CreateScalarStoreWithExtentsTest, Basic)
{
  const auto& extents = GetParam();
  auto runtime        = legate::Runtime::get_runtime();
  auto store          = runtime->create_store(legate::Scalar{SCALAR_VALUE}, extents);

  ASSERT_FALSE(store.unbound());
  ASSERT_EQ(store.dim(), extents.size());
  ASSERT_EQ(store.extents(), extents);
  ASSERT_EQ(store.volume(), 1);
  ASSERT_EQ(store.type().code(), legate::Type::Code::INT32);
  ASSERT_FALSE(store.transformed());
  ASSERT_TRUE(store.has_scalar_storage());
}

TEST_F(LogicalStoreCreateUnit, InvalidScalarStoreCreation)
{
  auto runtime = legate::Runtime::get_runtime();

  // volume > 1
  ASSERT_THROW(
    static_cast<void>(runtime->create_store(legate::Scalar{SCALAR_VALUE}, legate::Shape{1, 3})),
    std::invalid_argument);
}

TEST_F(LogicalStoreCreateUnit, BoundStoreToString)
{
  const auto runtime     = legate::Runtime::get_runtime();
  const auto bound_store = runtime->create_store(legate::Scalar{SCALAR_VALUE});

  ASSERT_THAT(
    bound_store.to_string(),
    ::testing::MatchesRegex(
      R"(Store\([0-9]+\) \{shape: \[1\], type: int32, storage: Storage\([0-9]+\) \{kind: Future, level: [0-9]+\}\})"));
}

TEST_F(LogicalStoreCreateUnit, TransformedBoundStoreToString)
{
  const auto runtime     = legate::Runtime::get_runtime();
  const auto bound_store = runtime->create_store(legate::Scalar{SCALAR_VALUE});
  const auto promoted    = bound_store.promote(0, 5);

  ASSERT_THAT(
    promoted.to_string(),
    ::testing::MatchesRegex(
      R"(Store\([0-9]+\) \{shape: \[5, 1\], transform: Promote\(extra_dim: 0, dim_size: 5\), type: int32, storage: Storage\([0-9]+\) \{kind: Future, level: [0-9]+\}\})"));
}

TEST_F(LogicalStoreCreateUnit, UnboundStoreToString)
{
  const auto runtime       = legate::Runtime::get_runtime();
  const auto unbound_store = runtime->create_store(legate::int64());

  ASSERT_THAT(
    unbound_store.to_string(),
    ::testing::MatchesRegex(
      R"(Store\([0-9]+\) \{shape: \(unbound\), type: int64, storage: Storage\([0-9]+\) \{kind: Region, level: [0-9]+, region: unbound\}\})"));
}

}  // namespace logical_store_create_test
