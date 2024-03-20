/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "legate.h"
#include "utilities/utilities.h"

#include <gtest/gtest.h>

namespace dispatch_test {

using DispatchTest = DefaultFixture;

constexpr std::array<legate::Type::Code, 14> PRIMITIVE_TYPE_CODE = {legate::Type::Code::BOOL,
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
                                                                    legate::Type::Code::COMPLEX128};

struct DoubleDispatchData {
  std::int32_t data1;
  std::int32_t data2;
};

struct double_dispatch_fn {
  template <legate::Type::Code CODE, std::int32_t DIM>
  void operator()(DoubleDispatchData& data)
  {
    EXPECT_EQ(CODE, static_cast<legate::Type::Code>(data.data1));
    EXPECT_EQ(DIM, data.data2);

    data.data1 = 1;
    data.data2 = 2;
    EXPECT_EQ(data.data1, 1);
    EXPECT_EQ(data.data2, 2);
  }
};

struct double_dispatch_with_dim_fn {
  template <std::int32_t DIM1, std::int32_t DIM2>
  void operator()(DoubleDispatchData& data)
  {
    EXPECT_EQ(DIM1, data.data1);
    EXPECT_EQ(DIM2, data.data2);

    data.data1 = 1;
    data.data2 = 2;
    EXPECT_EQ(data.data1, 1);
    EXPECT_EQ(data.data2, 2);
  }
};

struct dim_dispatch_fn {
  template <std::int32_t DIM>
  void operator()(legate::Scalar& scalar)
  {
    EXPECT_EQ(DIM, scalar.value<std::int32_t>());
  }
};

struct type_dispatch_fn {
  template <legate::Type::Code CODE>
  void operator()(legate::Scalar& scalar)
  {
    EXPECT_EQ(CODE, static_cast<legate::Type::Code>(scalar.value<std::uint32_t>()));
  }
};

TEST_F(DispatchTest, DoubleDispatch)
{
  for (std::size_t idx = 0; idx < PRIMITIVE_TYPE_CODE.size(); ++idx) {
    auto code               = PRIMITIVE_TYPE_CODE.at(idx);
    auto dim                = static_cast<std::int32_t>(idx % LEGATE_MAX_DIM + 1);
    DoubleDispatchData data = {static_cast<std::int32_t>(code), dim};
    legate::double_dispatch(dim, code, double_dispatch_fn{}, data);
  }

  // invalide dim
  DoubleDispatchData data = {1, 1};
  EXPECT_THROW(legate::double_dispatch(0, legate::Type::Code::BOOL, double_dispatch_fn{}, data),
               std::runtime_error);
  EXPECT_THROW(legate::double_dispatch(-1, legate::Type::Code::BOOL, double_dispatch_fn{}, data),
               std::runtime_error);
  EXPECT_THROW(legate::double_dispatch(
                 LEGATE_MAX_DIM + 1, legate::Type::Code::BOOL, double_dispatch_fn{}, data),
               std::runtime_error);

  // invalid type code
  EXPECT_THROW(
    legate::double_dispatch(1, legate::Type::Code::FIXED_ARRAY, double_dispatch_fn{}, data),
    std::runtime_error);
  EXPECT_THROW(legate::double_dispatch(1, legate::Type::Code::STRUCT, double_dispatch_fn{}, data),
               std::runtime_error);
  EXPECT_THROW(legate::double_dispatch(1, legate::Type::Code::STRING, double_dispatch_fn{}, data),
               std::runtime_error);
  EXPECT_THROW(legate::double_dispatch(1, legate::Type::Code::LIST, double_dispatch_fn{}, data),
               std::runtime_error);
  EXPECT_THROW(legate::double_dispatch(1, legate::Type::Code::NIL, double_dispatch_fn{}, data),
               std::runtime_error);
  EXPECT_THROW(legate::double_dispatch(1, legate::Type::Code::BINARY, double_dispatch_fn{}, data),
               std::runtime_error);
}

TEST_F(DispatchTest, DoubleDispatchWithDims)
{
  for (std::int32_t idx = 1; idx <= LEGATE_MAX_DIM; ++idx) {
    DoubleDispatchData data = {idx, LEGATE_MAX_DIM - idx + 1};
    legate::double_dispatch(idx, LEGATE_MAX_DIM - idx + 1, double_dispatch_with_dim_fn{}, data);
  }

  // invalid dim1
  DoubleDispatchData data = {1, 1};
  EXPECT_THROW(legate::double_dispatch(0, 1, double_dispatch_with_dim_fn{}, data),
               std::runtime_error);
  EXPECT_THROW(legate::double_dispatch(-1, 1, double_dispatch_with_dim_fn{}, data),
               std::runtime_error);
  EXPECT_THROW(legate::double_dispatch(LEGATE_MAX_DIM + 1, 1, double_dispatch_with_dim_fn{}, data),
               std::runtime_error);

  // invalid dim2
  EXPECT_THROW(legate::double_dispatch(1, 0, double_dispatch_with_dim_fn{}, data),
               std::runtime_error);
  EXPECT_THROW(legate::double_dispatch(1, -1, double_dispatch_with_dim_fn{}, data),
               std::runtime_error);
  EXPECT_THROW(legate::double_dispatch(1, LEGATE_MAX_DIM + 1, double_dispatch_with_dim_fn{}, data),
               std::runtime_error);
}

TEST_F(DispatchTest, DimDispatch)
{
  for (std::int32_t idx = 1; idx <= LEGATE_MAX_DIM; ++idx) {
    auto scalar = legate::Scalar{idx};

    legate::dim_dispatch(idx, dim_dispatch_fn{}, scalar);
  }

  // invalid dim
  auto scalar = legate::Scalar(1);
  EXPECT_THROW(legate::dim_dispatch(0, dim_dispatch_fn{}, scalar), std::runtime_error);
  EXPECT_THROW(legate::dim_dispatch(-1, dim_dispatch_fn{}, scalar), std::runtime_error);
  EXPECT_THROW(legate::dim_dispatch(LEGATE_MAX_DIM + 1, dim_dispatch_fn{}, scalar),
               std::runtime_error);
}

TEST_F(DispatchTest, TypeDispatch)
{
  for (auto code : PRIMITIVE_TYPE_CODE) {
    auto scalar = legate::Scalar{static_cast<std::uint32_t>(code)};

    legate::type_dispatch(code, type_dispatch_fn{}, scalar);
  }

  // invalid type code
  auto scalar = legate::Scalar(1);
  EXPECT_THROW(legate::type_dispatch(legate::Type::Code::FIXED_ARRAY, type_dispatch_fn{}, scalar),
               std::runtime_error);
  EXPECT_THROW(legate::type_dispatch(legate::Type::Code::STRUCT, type_dispatch_fn{}, scalar),
               std::runtime_error);
  EXPECT_THROW(legate::type_dispatch(legate::Type::Code::STRING, type_dispatch_fn{}, scalar),
               std::runtime_error);
  EXPECT_THROW(legate::type_dispatch(legate::Type::Code::LIST, type_dispatch_fn{}, scalar),
               std::runtime_error);
  EXPECT_THROW(legate::type_dispatch(legate::Type::Code::NIL, type_dispatch_fn{}, scalar),
               std::runtime_error);
  EXPECT_THROW(legate::type_dispatch(legate::Type::Code::BINARY, type_dispatch_fn{}, scalar),
               std::runtime_error);
}
}  // namespace dispatch_test
