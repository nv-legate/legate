/* Copyright 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <gtest/gtest.h>
#include "legate.h"

namespace span_test {


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
constexpr double DOUBLE_VALUE   = -4.567d;
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

TEST(SpanUnit, Create)
{
  create(BOOL_VALUE, BOOL_VALUE, !BOOL_VALUE);
  create(INT8_VALUE, static_cast<int8_t>(INT8_VALUE + 1), static_cast<int8_t>(INT8_VALUE + 2));
  create(INT16_VALUE, static_cast<int16_t>(INT16_VALUE + 1), static_cast<int16_t>(INT16_VALUE + 2));
  create(INT32_VALUE, static_cast<int32_t>(INT32_VALUE + 1), static_cast<int32_t>(INT32_VALUE + 2));
  create(INT64_VALUE, static_cast<int64_t>(INT64_VALUE + 1), static_cast<int64_t>(INT64_VALUE + 2));
  create(UINT8_VALUE, static_cast<uint8_t>(UINT8_VALUE + 1), static_cast<uint8_t>(UINT8_VALUE + 2));
  create(UINT16_VALUE, static_cast<uint16_t>(UINT16_VALUE + 1), static_cast<uint16_t>(UINT16_VALUE + 2));
  create(UINT32_VALUE, static_cast<uint32_t>(UINT32_VALUE + 1), static_cast<uint32_t>(UINT32_VALUE + 2));
  create(UINT64_VALUE, static_cast<uint64_t>(UINT64_VALUE + 1), static_cast<uint64_t>(UINT64_VALUE + 2));
  create(FLOAT_VALUE, FLOAT_VALUE + 1.0f, FLOAT_VALUE + 2.0f);
  create(DOUBLE_VALUE, DOUBLE_VALUE + 1.0d, DOUBLE_VALUE + 2.0d);
  create(STRING_VALUE, STRING_VALUE + "1", STRING_VALUE + "2");
  create(FLOAT16_VALUE1, FLOAT16_VALUE2, FLOAT16_VALUE3);
  create(COMPLEX_FLOAT_VALUE1, COMPLEX_FLOAT_VALUE2, COMPLEX_FLOAT_VALUE3);
  create(COMPLEX_DOUBLE_VALUE1, COMPLEX_DOUBLE_VALUE2, COMPLEX_DOUBLE_VALUE3);
}

TEST(SpanUnit, Subspan)
{
  auto data_vec        = std::vector<uint64_t>(DATA_SIZE, UINT64_VALUE);
  const auto* data     = data_vec.data();
  legate::Span span    = legate::Span<const uint64_t>(data, DATA_SIZE);
  legate::Span subspan = span.subspan(DATA_SIZE - 1);
  EXPECT_EQ(subspan.ptr(), &data[DATA_SIZE - 1]);
  EXPECT_EQ(subspan.size(), 1);
  EXPECT_EQ(subspan.end() - subspan.begin(), 1);
  for (auto& to_compare : span) EXPECT_EQ(UINT64_VALUE, to_compare);
}
}  // namespace span_test
