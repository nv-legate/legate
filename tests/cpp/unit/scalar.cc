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

#include "core/data/detail/scalar.h"
#include "core/utilities/deserializer.h"
#include "core/utilities/detail/buffer_builder.h"

namespace scalar_test {

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
const __half FLOAT16_VALUE(0.1f);
const complex<float> COMPLEX_FLOAT_VALUE{0, 1};
const complex<double> COMPLEX_DOUBLE_VALUE{0, 1};
constexpr uint32_t DATA_SIZE = 10;

struct PaddingStructData {
  bool bool_data;
  int32_t int32_data;
  uint64_t uint64_data;
};

struct __attribute__((packed)) NoPaddingStructData {
  bool bool_data;
  int32_t int32_data;
  uint64_t uint64_data;
};

class ScalarUnitTestDeserializer : public legate::BaseDeserializer<ScalarUnitTestDeserializer> {
 public:
  ScalarUnitTestDeserializer(const void* args, size_t arglen);

 public:
  using BaseDeserializer::_unpack;
};

ScalarUnitTestDeserializer::ScalarUnitTestDeserializer(const void* args, size_t arglen)
  : BaseDeserializer(args, arglen)
{
}

TEST(ScalarUnit, CreateWithObject)
{
  // constructor with Scalar object
  legate::Scalar scalar1(INT32_VALUE);
  legate::Scalar scalar2(scalar1);
  EXPECT_EQ(scalar2.type().code(), scalar1.type().code());
  EXPECT_EQ(scalar2.size(), scalar1.size());
  EXPECT_EQ(scalar2.size(), legate::uint32().size());
  EXPECT_NE(scalar2.ptr(), nullptr);
  EXPECT_EQ(scalar2.value<int32_t>(), scalar1.value<int32_t>());
  EXPECT_EQ(scalar2.values<int32_t>().size(), scalar1.values<int32_t>().size());
  EXPECT_EQ(*scalar2.values<int32_t>().begin(), *scalar1.values<int32_t>().begin());

  // Invalid type
  EXPECT_THROW(scalar2.value<int64_t>(), std::invalid_argument);
  EXPECT_THROW(scalar2.values<int8_t>(), std::invalid_argument);
}

TEST(ScalarUnit, CreateSharedScalar)
{
  auto data_vec    = std::vector<uint64_t>(DATA_SIZE, UINT64_VALUE);
  const auto* data = data_vec.data();
  legate::Scalar scalar(legate::uint64(), data);
  EXPECT_EQ(scalar.type().code(), legate::Type::Code::UINT64);
  EXPECT_EQ(scalar.size(), legate::uint64().size());
  EXPECT_EQ(scalar.ptr(), data);

  EXPECT_EQ(scalar.value<uint64_t>(), UINT64_VALUE);
  legate::Span actualValues(scalar.values<uint64_t>());
  legate::Span expectedValues = legate::Span<const uint64_t>(data, 1);
  EXPECT_EQ(*actualValues.begin(), *expectedValues.begin());
  EXPECT_EQ(actualValues.size(), expectedValues.size());

  // Invalid type
  EXPECT_THROW(scalar.value<int32_t>(), std::invalid_argument);
  EXPECT_THROW(scalar.values<int8_t>(), std::invalid_argument);
}

template <typename T>
void checkType(T value)
{
  legate::Scalar scalar(value);
  EXPECT_EQ(scalar.type().code(), legate::legate_type_code_of<T>);
  EXPECT_EQ(scalar.size(), sizeof(T));
  EXPECT_EQ(scalar.value<T>(), value);
  EXPECT_EQ(scalar.values<T>().size(), 1);
  EXPECT_NE(scalar.ptr(), nullptr);
}

TEST(ScalarUnit, CreateWithValue)
{
  checkType(BOOL_VALUE);
  checkType(INT8_VALUE);
  checkType(INT16_VALUE);
  checkType(INT32_VALUE);
  checkType(INT64_VALUE);
  checkType(UINT8_VALUE);
  checkType(UINT16_VALUE);
  checkType(UINT32_VALUE);
  checkType(UINT64_VALUE);
  checkType(FLOAT_VALUE);
  checkType(DOUBLE_VALUE);
  checkType(FLOAT16_VALUE);
  checkType(COMPLEX_FLOAT_VALUE);
  checkType(COMPLEX_DOUBLE_VALUE);
}

TEST(ScalarUnit, CreateWithVector)
{
  // constructor with arrays
  std::vector<int32_t> scalar_data{INT32_VALUE, INT32_VALUE};
  legate::Scalar scalar(scalar_data);
  EXPECT_EQ(scalar.type().code(), legate::Type::Code::FIXED_ARRAY);
  auto fixed_type = legate::fixed_array_type(legate::int32(), scalar_data.size());
  EXPECT_EQ(scalar.size(), fixed_type.size());

  // check values here. Note: no value allowed for a fixed arrays scalar
  std::vector<int32_t> data_vec = {INT32_VALUE, INT32_VALUE};
  const auto* data              = data_vec.data();
  legate::Span expectedValues   = legate::Span<const int32_t>(data, scalar_data.size());
  legate::Span actualValues(scalar.values<int32_t>());
  EXPECT_EQ(*actualValues.begin(), *expectedValues.begin());
  EXPECT_EQ(*actualValues.end(), *expectedValues.end());
  EXPECT_EQ(actualValues.size(), expectedValues.size());

  // Invalid type
  EXPECT_THROW(scalar.value<int32_t>(), std::invalid_argument);
  EXPECT_THROW(scalar.values<std::string>(), std::invalid_argument);
}

TEST(ScalarUnit, CreateWithString)
{
  // constructor with string
  auto inputString = STRING_VALUE;
  legate::Scalar scalar(inputString);
  EXPECT_EQ(scalar.type().code(), legate::Type::Code::STRING);
  auto expectedSize = sizeof(uint32_t) + sizeof(char) * inputString.size();
  EXPECT_EQ(scalar.size(), expectedSize);
  EXPECT_NE(scalar.ptr(), inputString.data());

  // Check values
  EXPECT_EQ(scalar.value<std::string>(), inputString);
  EXPECT_EQ(scalar.value<std::string_view>(), inputString);
  legate::Span actualValues(scalar.values<char>());
  EXPECT_EQ(actualValues.size(), inputString.size());
  EXPECT_EQ(*actualValues.begin(), inputString[0]);

  // Invalid type
  EXPECT_THROW(scalar.value<int32_t>(), std::invalid_argument);
  EXPECT_THROW(scalar.values<int32_t>(), std::invalid_argument);
}

TEST(ScalarUnit, CreateWithStructType)
{
  // with struct padding
  {
    PaddingStructData structData = {BOOL_VALUE, INT32_VALUE, UINT64_VALUE};
    legate::Scalar scalar(
      structData, legate::struct_type(true, legate::bool_(), legate::int32(), legate::uint64()));
    EXPECT_EQ(scalar.type().code(), legate::Type::Code::STRUCT);
    EXPECT_EQ(scalar.size(), sizeof(PaddingStructData));
    EXPECT_NE(scalar.ptr(), nullptr);

    // Check value
    PaddingStructData actualData = scalar.value<PaddingStructData>();
    EXPECT_EQ(actualData.bool_data, structData.bool_data);
    EXPECT_EQ(actualData.int32_data, structData.int32_data);
    EXPECT_EQ(actualData.uint64_data, structData.uint64_data);

    // Check values
    legate::Span actualValues(scalar.values<PaddingStructData>());
    legate::Span expectedValues = legate::Span<const PaddingStructData>(&structData, 1);
    EXPECT_EQ(actualValues.size(), expectedValues.size());
    EXPECT_EQ(actualValues.begin()->bool_data, expectedValues.begin()->bool_data);
    EXPECT_EQ(actualValues.begin()->int32_data, expectedValues.begin()->int32_data);
    EXPECT_EQ(actualValues.begin()->uint64_data, expectedValues.begin()->uint64_data);
    EXPECT_NE(actualValues.ptr(), expectedValues.ptr());
  }

  // without struct padding
  {
    NoPaddingStructData structData = {BOOL_VALUE, INT32_VALUE, UINT64_VALUE};
    legate::Scalar scalar(
      structData, legate::struct_type(false, legate::bool_(), legate::int32(), legate::uint64()));
    EXPECT_EQ(scalar.type().code(), legate::Type::Code::STRUCT);
    EXPECT_EQ(scalar.size(), sizeof(NoPaddingStructData));
    EXPECT_NE(scalar.ptr(), nullptr);

    // Check value
    NoPaddingStructData actualData = scalar.value<NoPaddingStructData>();
    EXPECT_EQ(actualData.bool_data, structData.bool_data);
    EXPECT_EQ(actualData.int32_data, structData.int32_data);
    EXPECT_EQ(actualData.uint64_data, structData.uint64_data);

    // Check values
    legate::Span actualValues(scalar.values<NoPaddingStructData>());
    legate::Span expectedValues = legate::Span<const NoPaddingStructData>(&structData, 1);
    EXPECT_EQ(actualValues.size(), expectedValues.size());
    EXPECT_EQ(actualValues.begin()->bool_data, expectedValues.begin()->bool_data);
    EXPECT_EQ(actualValues.begin()->int32_data, expectedValues.begin()->int32_data);
    EXPECT_EQ(actualValues.begin()->uint64_data, expectedValues.begin()->uint64_data);
    EXPECT_NE(actualValues.ptr(), expectedValues.ptr());
  }
}

TEST(ScalarUnit, OperatorEqual)
{
  legate::Scalar scalar1(INT32_VALUE);
  legate::Scalar scalar2(UINT64_VALUE);
  scalar2 = scalar1;
  EXPECT_EQ(scalar2.type().code(), scalar1.type().code());
  EXPECT_EQ(scalar2.size(), scalar2.size());
  EXPECT_EQ(scalar2.value<int32_t>(), scalar1.value<int32_t>());
  EXPECT_EQ(scalar2.values<int32_t>().size(), scalar1.values<int32_t>().size());
}

void checkPack(const legate::Scalar& scalar)
{
  legate::detail::BufferBuilder buf;
  scalar.impl()->pack(buf);
  auto legion_buffer = buf.to_legion_buffer();
  EXPECT_NE(legion_buffer.get_ptr(), nullptr);
  legate::BaseDeserializer<ScalarUnitTestDeserializer> deserializer(legion_buffer.get_ptr(),
                                                                    legion_buffer.get_size());
  auto scalar_unpack = deserializer._unpack_scalar();
  EXPECT_EQ(scalar_unpack.type().code(), scalar.type().code());
  EXPECT_EQ(scalar_unpack.size(), scalar.size());
}

TEST(ScalarUnit, Pack)
{
  // test pack for a fixed array scalar
  {
    legate::Scalar scalar(std::vector<uint32_t>{UINT32_VALUE, UINT32_VALUE});
    checkPack(scalar);
  }

  // test pack for a single value scalar
  {
    legate::Scalar scalar(BOOL_VALUE);
    checkPack(scalar);
  }

  // test pack for string scalar
  {
    legate::Scalar scalar(STRING_VALUE);
    checkPack(scalar);
  }

  // test pack for padding struct type scalar
  {
    PaddingStructData structData = {BOOL_VALUE, INT32_VALUE, UINT64_VALUE};
    legate::Scalar scalar(
      structData, legate::struct_type(true, legate::bool_(), legate::int32(), legate::uint64()));
    checkPack(scalar);
  }

  // test pack for no padding struct type scalar
  {
    NoPaddingStructData structData = {BOOL_VALUE, INT32_VALUE, UINT64_VALUE};
    legate::Scalar scalar(
      structData, legate::struct_type(false, legate::bool_(), legate::int32(), legate::uint64()));
    checkPack(scalar);
  }

  // test pack for a shared scalar
  {
    auto data_vec    = std::vector<uint64_t>(DATA_SIZE, UINT64_VALUE);
    const auto* data = data_vec.data();
    legate::Scalar scalar(legate::uint64(), data);
    checkPack(scalar);
  }

  // test pack for scalar created by another scalar
  {
    legate::Scalar scalar1(INT32_VALUE);
    legate::Scalar scalar2(scalar1);
    checkPack(scalar2);
  }
}
}  // namespace scalar_test
