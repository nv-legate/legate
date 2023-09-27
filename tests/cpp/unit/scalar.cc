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
constexpr double DOUBLE_VALUE   = -4.567;
const std::string STRING_VALUE  = "123";
const __half FLOAT16_VALUE(0.1f);
const complex<float> COMPLEX_FLOAT_VALUE{0, 1};
const complex<double> COMPLEX_DOUBLE_VALUE{0, 1};
constexpr uint32_t DATA_SIZE = 10;

struct PaddingStructData {
  bool bool_data;
  int32_t int32_data;
  uint64_t uint64_data;
  bool operator==(const PaddingStructData& other) const
  {
    return bool_data == other.bool_data && int32_data == other.int32_data &&
           uint64_data == other.uint64_data;
  }
};

struct __attribute__((packed)) NoPaddingStructData {
  bool bool_data;
  int32_t int32_data;
  uint64_t uint64_data;
  bool operator==(const NoPaddingStructData& other) const
  {
    return bool_data == other.bool_data && int32_data == other.int32_data &&
           uint64_data == other.uint64_data;
  }
};

template <int32_t DIM>
struct MultiDimRectStruct {
  int64_t lo[DIM];
  int64_t hi[DIM];
};

template <typename T>
void check_type(T value);
template <typename T>
void check_struct_type_scalar(T& struct_data, bool align);
template <int32_t DIM>
void check_point_scalar(const int64_t bounds[]);
template <int32_t DIM>
void check_rect_scalar(const int64_t lo[], const int64_t hi[]);
template <int32_t DIM>
void check_rect_bounds(const MultiDimRectStruct<DIM>& to_compare,
                       const MultiDimRectStruct<DIM>& expect);
void check_pack(const legate::Scalar& scalar);
template <int32_t DIM>
void check_pack_point_scalar();
template <int32_t DIM>
void check_pack_rect_scalar();

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

template <typename T>
void check_type(T value, legate::Type type)
{
  {
    legate::Scalar scalar(value);
    EXPECT_EQ(scalar.type().code(), legate::legate_type_code_of<T>);
    EXPECT_EQ(scalar.size(), sizeof(T));
    EXPECT_EQ(scalar.value<T>(), value);
    EXPECT_EQ(scalar.values<T>().size(), 1);
    EXPECT_NE(scalar.ptr(), nullptr);
  }

  {
    legate::Scalar scalar(value, type);
    EXPECT_EQ(scalar.type().code(), type.code());
    EXPECT_EQ(scalar.size(), type.size());
    EXPECT_EQ(scalar.value<T>(), value);
    EXPECT_EQ(scalar.values<T>().size(), 1);
    EXPECT_NE(scalar.ptr(), nullptr);
  }
}

template <typename T>
void check_binary_scalar(T& value)
{
  legate::Scalar scalar(value, legate::binary_type(sizeof(T)));
  EXPECT_EQ(scalar.type().size(), sizeof(T));
  EXPECT_NE(scalar.ptr(), nullptr);

  EXPECT_EQ(value, scalar.value<T>());
  auto actual = scalar.values<T>();
  EXPECT_EQ(actual.size(), 1);
  EXPECT_EQ(actual[0], value);
}

template <typename T>
void check_struct_type_scalar(T& struct_data, bool align)
{
  legate::Scalar scalar(
    struct_data, legate::struct_type(align, legate::bool_(), legate::int32(), legate::uint64()));
  EXPECT_EQ(scalar.type().code(), legate::Type::Code::STRUCT);
  EXPECT_EQ(scalar.size(), sizeof(T));
  EXPECT_NE(scalar.ptr(), nullptr);

  // Check value
  T actual_data = scalar.value<T>();
  EXPECT_EQ(actual_data.bool_data, struct_data.bool_data);
  EXPECT_EQ(actual_data.int32_data, struct_data.int32_data);
  EXPECT_EQ(actual_data.uint64_data, struct_data.uint64_data);

  // Check values
  legate::Span actual_values(scalar.values<T>());
  legate::Span expected_values = legate::Span<const T>(&struct_data, 1);
  EXPECT_EQ(actual_values.size(), expected_values.size());
  EXPECT_EQ(actual_values.begin()->bool_data, expected_values.begin()->bool_data);
  EXPECT_EQ(actual_values.begin()->int32_data, expected_values.begin()->int32_data);
  EXPECT_EQ(actual_values.begin()->uint64_data, expected_values.begin()->uint64_data);
  EXPECT_NE(actual_values.ptr(), expected_values.ptr());
}

template <int32_t DIM>
void check_point_scalar(const int64_t bounds[])
{
  auto point = legate::Point<DIM>(bounds);
  legate::Scalar scalar(point);
  EXPECT_EQ(scalar.type().code(), legate::Type::Code::FIXED_ARRAY);
  auto fixed_type = legate::fixed_array_type(legate::int64(), DIM);
  EXPECT_EQ(scalar.size(), fixed_type.size());
  EXPECT_NE(scalar.ptr(), nullptr);

  // Check values
  legate::Span expectedValues = legate::Span<const int64_t>(bounds, DIM);
  legate::Span actualValues(scalar.values<int64_t>());
  for (int i = 0; i < DIM; i++) { EXPECT_EQ(actualValues[i], expectedValues[i]); }
  EXPECT_EQ(actualValues.size(), DIM);
  EXPECT_EQ(actualValues.size(), expectedValues.size());

  // Invalid type
  EXPECT_THROW(scalar.values<int32_t>(), std::invalid_argument);
}

template <int32_t DIM>
void check_rect_scalar(const int64_t lo[], const int64_t hi[])
{
  auto point_lo = legate::Point<DIM>(lo);
  auto point_hi = legate::Point<DIM>(hi);
  auto rect     = legate::Rect<DIM>(point_lo, point_hi);
  legate::Scalar scalar(rect);
  EXPECT_EQ(scalar.type().code(), legate::Type::Code::STRUCT);
  auto struct_type = legate::struct_type(true,
                                         legate::fixed_array_type(legate::int64(), DIM),
                                         legate::fixed_array_type(legate::int64(), DIM));
  EXPECT_EQ(scalar.size(), struct_type.size());
  EXPECT_NE(scalar.ptr(), nullptr);

  // Check values
  MultiDimRectStruct<DIM> actual_data = scalar.value<MultiDimRectStruct<DIM>>();
  MultiDimRectStruct<DIM> expected_data;
  for (int i = 0; i < DIM; i++) {
    expected_data.lo[i] = lo[i];
    expected_data.hi[i] = hi[i];
  }
  check_rect_bounds(actual_data, expected_data);

  legate::Span actual_values(scalar.values<MultiDimRectStruct<DIM>>());
  legate::Span expected_values = legate::Span<const MultiDimRectStruct<DIM>>(&expected_data, 1);
  EXPECT_EQ(actual_values.size(), 1);
  EXPECT_EQ(actual_values.size(), expected_values.size());
  check_rect_bounds(*actual_values.begin(), *expected_values.begin());
  EXPECT_NE(actual_values.ptr(), expected_values.ptr());

  // Invalid type
  EXPECT_THROW(scalar.value<int32_t>(), std::invalid_argument);
  EXPECT_THROW(scalar.values<uint8_t>(), std::invalid_argument);
}

template <int32_t DIM>
void check_rect_bounds(const MultiDimRectStruct<DIM>& to_compare,
                       const MultiDimRectStruct<DIM>& expect)
{
  EXPECT_EQ(std::size(to_compare.lo), std::size(to_compare.hi));
  EXPECT_EQ(std::size(expect.lo), std::size(expect.hi));
  EXPECT_EQ(std::size(to_compare.lo), std::size(expect.lo));

  for (std::size_t i = 0; i < std::size(to_compare.lo); i++) {
    EXPECT_EQ(to_compare.lo[i], expect.lo[i]);
    EXPECT_EQ(to_compare.hi[i], expect.hi[i]);
  }
}

void check_pack(const legate::Scalar& scalar)
{
  legate::detail::BufferBuilder buf;
  scalar.impl()->pack(buf);
  auto legion_buffer = buf.to_legion_buffer();
  EXPECT_NE(legion_buffer.get_ptr(), nullptr);
  legate::BaseDeserializer<ScalarUnitTestDeserializer> deserializer(legion_buffer.get_ptr(),
                                                                    legion_buffer.get_size());
  auto scalar_unpack = deserializer.unpack_scalar();
  EXPECT_EQ(scalar_unpack->type()->code, scalar.type().code());
  EXPECT_EQ(scalar_unpack->size(), scalar.size());
}

template <int32_t DIM>
void check_pack_point_scalar()
{
  auto point = legate::Point<DIM>::ONES();
  legate::Scalar scalar(point);
  check_pack(scalar);
}

template <int32_t DIM>
void check_pack_rect_scalar()
{
  auto rect = legate::Rect<DIM>(legate::Point<DIM>::ZEROES(), legate::Point<DIM>::ONES());
  legate::Scalar scalar(rect);
  check_pack(scalar);
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
  // unowned
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

  // owned
  {
    auto data_vec    = std::vector<uint32_t>(DATA_SIZE, UINT32_VALUE);
    const auto* data = data_vec.data();
    legate::Scalar scalar(legate::uint32(), data, true);
    EXPECT_NE(scalar.ptr(), data);
  }
}

TEST(ScalarUnit, CreateWithValue)
{
  check_type(BOOL_VALUE, legate::bool_());
  check_type(INT8_VALUE, legate::int8());
  check_type(INT16_VALUE, legate::int16());
  check_type(INT32_VALUE, legate::int32());
  check_type(INT64_VALUE, legate::int64());
  check_type(UINT8_VALUE, legate::uint8());
  check_type(UINT16_VALUE, legate::uint16());
  check_type(UINT32_VALUE, legate::uint32());
  check_type(UINT64_VALUE, legate::uint64());
  check_type(FLOAT_VALUE, legate::float32());
  check_type(DOUBLE_VALUE, legate::float64());
  check_type(FLOAT16_VALUE, legate::float16());
  check_type(COMPLEX_FLOAT_VALUE, legate::complex64());
  check_type(COMPLEX_DOUBLE_VALUE, legate::complex128());

  // Size mismatch
  EXPECT_THROW(legate::Scalar(FLOAT16_VALUE, legate::float32()), std::invalid_argument);
}

TEST(ScalarUnit, CreateBinary)
{
  {
    PaddingStructData value = {BOOL_VALUE, INT32_VALUE, UINT64_VALUE};
    check_binary_scalar(value);
  }

  {
    NoPaddingStructData value = {BOOL_VALUE, INT32_VALUE, UINT64_VALUE};
    check_binary_scalar(value);
  }
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
  for (std::size_t i = 0; i < scalar_data.size(); i++) {
    EXPECT_EQ(actualValues[i], expectedValues[i]);
  }
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
    PaddingStructData struct_data = {BOOL_VALUE, INT32_VALUE, UINT64_VALUE};
    check_struct_type_scalar(struct_data, true);
  }

  // without struct padding
  {
    NoPaddingStructData struct_data = {BOOL_VALUE, INT32_VALUE, UINT64_VALUE};
    check_struct_type_scalar(struct_data, false);
  }
}

TEST(ScalarUnit, CreateWithPoint)
{
  {
    const int64_t bounds[] = {0};
    check_point_scalar<1>(bounds);
  }

  {
    const int64_t bounds[] = {1, 8};
    check_point_scalar<2>(bounds);
  }

  {
    const int64_t bounds[] = {-10, 2, -1};
    check_point_scalar<3>(bounds);
  }

  {
    const int64_t bounds[] = {1, 5, 7, 200};
    check_point_scalar<4>(bounds);
  }

  // Invalid dim
  EXPECT_THROW(legate::Scalar{legate::Point<10>::ONES()}, std::out_of_range);
}

TEST(ScalarUnit, CreateWithRect)
{
  {
    const int64_t lo[] = {1};
    const int64_t hi[] = {-9};
    check_rect_scalar<1>(lo, hi);
  }

  {
    const int64_t lo[] = {-1, 3};
    const int64_t hi[] = {9, -2};
    check_rect_scalar<2>(lo, hi);
  }

  {
    const int64_t lo[] = {0, 1, 2};
    const int64_t hi[] = {3, 4, 5};
    check_rect_scalar<3>(lo, hi);
  }

  {
    const int64_t lo[] = {-5, 1, -7, 10};
    const int64_t hi[] = {4, 5, 6, 7};
    check_rect_scalar<4>(lo, hi);
  }

  // Invalid dim
  EXPECT_THROW(
    legate::Scalar(legate::Rect<100>(legate::Point<100>::ZEROES(), legate::Point<100>::ZEROES())),
    std::out_of_range);
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

TEST(ScalarUnit, Pack)
{
  // test pack for a fixed array scalar
  {
    legate::Scalar scalar(std::vector<uint32_t>{UINT32_VALUE, UINT32_VALUE});
    check_pack(scalar);
  }

  // test pack for a single value scalar
  {
    legate::Scalar scalar(BOOL_VALUE);
    check_pack(scalar);
  }

  // test pack for string scalar
  {
    legate::Scalar scalar(STRING_VALUE);
    check_pack(scalar);
  }

  // test pack for padding struct type scalar
  {
    PaddingStructData structData = {BOOL_VALUE, INT32_VALUE, UINT64_VALUE};
    legate::Scalar scalar(
      structData, legate::struct_type(true, legate::bool_(), legate::int32(), legate::uint64()));
    check_pack(scalar);
  }

  // test pack for no padding struct type scalar
  {
    NoPaddingStructData structData = {BOOL_VALUE, INT32_VALUE, UINT64_VALUE};
    legate::Scalar scalar(
      structData, legate::struct_type(false, legate::bool_(), legate::int32(), legate::uint64()));
    check_pack(scalar);
  }

  // test pack for a shared scalar
  {
    auto data_vec    = std::vector<uint64_t>(DATA_SIZE, UINT64_VALUE);
    const auto* data = data_vec.data();
    legate::Scalar scalar(legate::uint64(), data);
    check_pack(scalar);
  }

  // test pack for scalar created by another scalar
  {
    legate::Scalar scalar1(INT32_VALUE);
    legate::Scalar scalar2(scalar1);
    check_pack(scalar2);
  }

  // test pack for scalar created by point
  {
    check_pack_point_scalar<1>();
    check_pack_point_scalar<2>();
    check_pack_point_scalar<3>();
    check_pack_point_scalar<4>();
  }

  // test pack for scalar created by rect
  {
    check_pack_rect_scalar<1>();
    check_pack_rect_scalar<2>();
    check_pack_rect_scalar<3>();
    check_pack_rect_scalar<4>();
  }
}
}  // namespace scalar_test
