/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace dispatch_test {

using DispatchTest = DefaultFixture;

class DoubleDispatchTest
  : public DefaultFixture,
    public ::testing::WithParamInterface<std::tuple<int, legate::Type::Code>> {};

class DoubleDispatchWithDimTest : public DefaultFixture,
                                  public ::testing::WithParamInterface<std::tuple<int, int>> {};

class DimDispatchTest : public DefaultFixture, public ::testing::WithParamInterface<int> {};

class TypeDispatchTest : public DefaultFixture,
                         public ::testing::WithParamInterface<legate::Type::Code> {};

class DispatchNegativeDimTest : public DefaultFixture, public ::testing::WithParamInterface<int> {};

class DispatchNegativeTypeTest : public DefaultFixture,
                                 public ::testing::WithParamInterface<legate::Type::Code> {};

INSTANTIATE_TEST_SUITE_P(DispatchTest,
                         DoubleDispatchTest,
                         ::testing::Combine(::testing::Range(1, LEGATE_MAX_DIM),
                                            ::testing::Values(legate::Type::Code::BOOL,
                                                              legate::Type::Code::INT8,
                                                              legate::Type::Code::INT16,
                                                              legate::Type::Code::INT32,
                                                              legate::Type::Code::INT64,
                                                              legate::Type::Code::UINT8,
                                                              legate::Type::Code::UINT16,
                                                              legate::Type::Code::UINT32,
                                                              legate::Type::Code::UINT64,
                                                              legate::Type::Code::FLOAT16,
                                                              legate::Type::Code::FLOAT32,
                                                              legate::Type::Code::FLOAT64,
                                                              legate::Type::Code::COMPLEX64,
                                                              legate::Type::Code::COMPLEX128)));

INSTANTIATE_TEST_SUITE_P(DispatchTest,
                         DoubleDispatchWithDimTest,
                         ::testing::Combine(::testing::Range(1, LEGATE_MAX_DIM),
                                            ::testing::Range(1, LEGATE_MAX_DIM)));

INSTANTIATE_TEST_SUITE_P(DispatchTest, DimDispatchTest, ::testing::Range(1, LEGATE_MAX_DIM));

INSTANTIATE_TEST_SUITE_P(DispatchTest,
                         TypeDispatchTest,
                         ::testing::Values(legate::Type::Code::BOOL,
                                           legate::Type::Code::INT8,
                                           legate::Type::Code::INT16,
                                           legate::Type::Code::INT32,
                                           legate::Type::Code::INT64,
                                           legate::Type::Code::UINT8,
                                           legate::Type::Code::UINT16,
                                           legate::Type::Code::UINT32,
                                           legate::Type::Code::UINT64,
                                           legate::Type::Code::FLOAT16,
                                           legate::Type::Code::FLOAT32,
                                           legate::Type::Code::FLOAT64,
                                           legate::Type::Code::COMPLEX64,
                                           legate::Type::Code::COMPLEX128));

INSTANTIATE_TEST_SUITE_P(DispatchTest,
                         DispatchNegativeDimTest,
                         ::testing::Values(0, -1, LEGATE_MAX_DIM + 1));

INSTANTIATE_TEST_SUITE_P(DispatchTest,
                         DispatchNegativeTypeTest,
                         ::testing::Values(legate::Type::Code::FIXED_ARRAY,
                                           legate::Type::Code::STRUCT,
                                           legate::Type::Code::STRING,
                                           legate::Type::Code::LIST,
                                           legate::Type::Code::NIL,
                                           legate::Type::Code::BINARY));

class DoubleDispatchFn {
 public:
  template <legate::Type::Code CODE, std::int32_t DIM>
  void operator()(legate::Type::Code code, std::int32_t dim)
  {
    ASSERT_EQ(CODE, code);
    ASSERT_EQ(DIM, dim);
  }
};

class DoubleDispatchWithDimFn {
 public:
  template <std::int32_t DIM1, std::int32_t DIM2>
  void operator()(std::int32_t dim1, std::int32_t dim2)
  {
    ASSERT_EQ(DIM1, dim1);
    ASSERT_EQ(DIM2, dim2);
  }
};

class DimDispatchFn {
 public:
  template <std::int32_t DIM>
  void operator()(legate::Scalar& scalar)
  {
    ASSERT_EQ(DIM, scalar.value<std::int32_t>());
  }
};

class TypeDispatchFn {
 public:
  template <legate::Type::Code CODE>
  void operator()(legate::Scalar& scalar)
  {
    ASSERT_EQ(CODE, static_cast<legate::Type::Code>(scalar.value<std::uint32_t>()));
  }
};

TEST_P(DoubleDispatchTest, DoubleDispatch)
{
  const auto [dim, code] = GetParam();

  legate::double_dispatch(dim, code, DoubleDispatchFn{}, code, dim);
}

TEST_P(DispatchNegativeDimTest, DoubleDispatch)
{
  const auto dim = GetParam();

  ASSERT_THROW(
    legate::double_dispatch(
      dim, legate::Type::Code::BOOL, DoubleDispatchFn{}, legate::Type::Code::BOOL, LEGATE_MAX_DIM),
    std::runtime_error);
}

TEST_P(DispatchNegativeTypeTest, DoubleDispatch)
{
  const auto type = GetParam();

  ASSERT_THROW(
    legate::double_dispatch(
      LEGATE_MAX_DIM, type, DoubleDispatchFn{}, legate::Type::Code::BOOL, LEGATE_MAX_DIM),
    std::runtime_error);
}

TEST_P(DoubleDispatchWithDimTest, DoubleDispatchWithDim)
{
  const auto [dim1, dim2] = GetParam();

  legate::double_dispatch(dim1, dim2, DoubleDispatchWithDimFn{}, dim1, dim2);
}

TEST_P(DispatchNegativeDimTest, DoubleDispatchWithDim)
{
  const auto dim = GetParam();

  // invalid dim1
  ASSERT_THROW(
    legate::double_dispatch(dim, LEGATE_MAX_DIM, DoubleDispatchWithDimFn{}, dim, LEGATE_MAX_DIM),
    std::runtime_error);

  // invalid dim2
  ASSERT_THROW(
    legate::double_dispatch(LEGATE_MAX_DIM, dim, DoubleDispatchWithDimFn{}, LEGATE_MAX_DIM, dim),
    std::runtime_error);
}

TEST_P(DimDispatchTest, DimDispatch)
{
  const auto dim = GetParam();
  auto scalar    = legate::Scalar{static_cast<std::uint32_t>(dim)};

  legate::dim_dispatch(dim, DimDispatchFn{}, scalar);
}

TEST_P(DispatchNegativeDimTest, DimDispatch)
{
  const auto dim = GetParam();
  auto scalar    = legate::Scalar{static_cast<std::uint32_t>(dim)};

  ASSERT_THROW(legate::dim_dispatch(dim, DimDispatchFn{}, scalar), std::runtime_error);
}

TEST_P(TypeDispatchTest, TypeDispatch)
{
  const auto type = GetParam();
  auto scalar     = legate::Scalar{static_cast<std::uint32_t>(type)};

  legate::type_dispatch(type, TypeDispatchFn{}, scalar);
}

TEST_P(DispatchNegativeTypeTest, TypeDispatch)
{
  const auto type = GetParam();
  auto scalar     = legate::Scalar{static_cast<std::uint32_t>(type)};

  ASSERT_THROW(legate::type_dispatch(type, TypeDispatchFn{}, scalar), std::runtime_error);
}

}  // namespace dispatch_test
