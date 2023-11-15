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

namespace span_test {

using SpanUnit = DefaultFixture;

constexpr bool BOOL_VALUE       = true;
constexpr int8_t INT8_VALUE     = 10;
constexpr int16_t INT16_VALUE   = -1000;
constexpr int32_t INT32_VALUE   = 2700;
constexpr int64_t INT64_VALUE   = -28;
constexpr uint8_t UINT8_VALUE   = 128;
constexpr uint16_t UINT16_VALUE = 65535;
constexpr uint32_t UINT32_VALUE = 999;
constexpr uint64_t UINT64_VALUE = 100;
constexpr float FLOAT_VALUE     = 1.23f;
constexpr double DOUBLE_VALUE   = -4.567;
const std::string STRING_VALUE  = "123";
const __half FLOAT16_VALUE1(0.1f);
const __half FLOAT16_VALUE2(0.2f);
const __half FLOAT16_VALUE3(0.3f);
const complex<float> COMPLEX_FLOAT_VALUE1{0, 1};
const complex<float> COMPLEX_FLOAT_VALUE2{2, 3};
const complex<float> COMPLEX_FLOAT_VALUE3{4, 5};
const complex<double> COMPLEX_DOUBLE_VALUE1{6, 7};
const complex<double> COMPLEX_DOUBLE_VALUE2{8, 9};
const complex<double> COMPLEX_DOUBLE_VALUE3{10, 11};

constexpr uint32_t DATA_SIZE = 3;

template <typename T>
void create(T value1, T value2, T value3)
{
  std::array<T, DATA_SIZE> data = {value1, value2, value3};
  legate::Span span             = legate::Span<const T>(data.data(), DATA_SIZE);
  EXPECT_EQ(span.ptr(), data.data());
  EXPECT_EQ(span.size(), DATA_SIZE);
  EXPECT_EQ(span.end() - span.begin(), DATA_SIZE);
  for (auto& to_compare : span) {
    auto i = std::distance(span.begin(), &to_compare);
    EXPECT_EQ(data.at(i), to_compare);
  }
  EXPECT_EQ(span[0], value1);
  EXPECT_EQ(span[DATA_SIZE - 1], value3);
}

TEST_F(SpanUnit, Create)
{
  create(BOOL_VALUE, BOOL_VALUE, !BOOL_VALUE);
  create(INT8_VALUE, static_cast<int8_t>(INT8_VALUE + 1), static_cast<int8_t>(INT8_VALUE + 2));
  create(INT16_VALUE, static_cast<int16_t>(INT16_VALUE + 1), static_cast<int16_t>(INT16_VALUE + 2));
  create(INT32_VALUE, static_cast<int32_t>(INT32_VALUE + 1), static_cast<int32_t>(INT32_VALUE + 2));
  create(INT64_VALUE, static_cast<int64_t>(INT64_VALUE + 1), static_cast<int64_t>(INT64_VALUE + 2));
  create(UINT8_VALUE, static_cast<uint8_t>(UINT8_VALUE + 1), static_cast<uint8_t>(UINT8_VALUE + 2));
  create(
    UINT16_VALUE, static_cast<uint16_t>(UINT16_VALUE + 1), static_cast<uint16_t>(UINT16_VALUE + 2));
  create(
    UINT32_VALUE, static_cast<uint32_t>(UINT32_VALUE + 1), static_cast<uint32_t>(UINT32_VALUE + 2));
  create(
    UINT64_VALUE, static_cast<uint64_t>(UINT64_VALUE + 1), static_cast<uint64_t>(UINT64_VALUE + 2));
  create(FLOAT_VALUE, FLOAT_VALUE + 1.0f, FLOAT_VALUE + 2.0f);
  create(DOUBLE_VALUE, DOUBLE_VALUE + 1.0, DOUBLE_VALUE + 2.0);
  create(STRING_VALUE, STRING_VALUE + "1", STRING_VALUE + "2");
  create(FLOAT16_VALUE1, FLOAT16_VALUE2, FLOAT16_VALUE3);
  create(COMPLEX_FLOAT_VALUE1, COMPLEX_FLOAT_VALUE2, COMPLEX_FLOAT_VALUE3);
  create(COMPLEX_DOUBLE_VALUE1, COMPLEX_DOUBLE_VALUE2, COMPLEX_DOUBLE_VALUE3);
}

TEST_F(SpanUnit, Subspan)
{
  auto data_vec        = std::vector<uint64_t>(DATA_SIZE, UINT64_VALUE);
  const auto* data     = data_vec.data();
  legate::Span span    = legate::Span<const uint64_t>(data, DATA_SIZE);
  legate::Span subspan = span.subspan(DATA_SIZE - 1);
  EXPECT_EQ(subspan.ptr(), &data[DATA_SIZE - 1]);
  EXPECT_EQ(subspan.size(), 1);
  EXPECT_EQ(subspan.end() - subspan.begin(), 1);
  for (auto& to_compare : span) {
    EXPECT_EQ(UINT64_VALUE, to_compare);
  }
}
}  // namespace span_test
