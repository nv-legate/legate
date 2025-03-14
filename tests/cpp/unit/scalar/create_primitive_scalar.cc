/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/scalar.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace create_primitive_scalar_test {

namespace {

using PrimitiveScalarUnit = DefaultFixture;

constexpr bool BOOL_VALUE            = true;
constexpr std::int8_t INT8_VALUE     = 10;
constexpr std::int16_t INT16_VALUE   = -1000;
constexpr std::int32_t INT32_VALUE   = 2700;
constexpr std::int64_t INT64_VALUE   = -28;
constexpr std::uint8_t UINT8_VALUE   = 128;
constexpr std::uint16_t UINT16_VALUE = 65535;
constexpr std::uint32_t UINT32_VALUE = 999;
constexpr std::uint64_t UINT64_VALUE = 100;
constexpr float FLOAT_VALUE          = 1.23F;
constexpr double DOUBLE_VALUE        = -4.567;
// NOLINTBEGIN(cert-err58-cpp)
const std::string STRING_VALUE = "123";
const __half FLOAT16_VALUE(0.1F);
const complex<float> COMPLEX_FLOAT_VALUE{0, 1};
const complex<double> COMPLEX_DOUBLE_VALUE{0, 1};
// NOLINTEND(cert-err58-cpp)
constexpr std::uint32_t DATA_SIZE = 10;

enum class SmallEnumType : std::uint8_t { FOO };
enum class BigEnumType : std::int64_t {
  // These values are there only to ensure that clang-tidy does not complain about using too
  // small an underlying type for the enum
  FOO = std::numeric_limits<std::int64_t>::max(),
  BAR = std::numeric_limits<std::int64_t>::min()
};

template <std::int32_t DIM>
struct MultiDimRectStruct {
  std::int64_t lo[DIM];
  std::int64_t hi[DIM];
};

class PrimitiveScalarTest
  : public PrimitiveScalarUnit,
    public ::testing::WithParamInterface<std::tuple<legate::Scalar, legate::Type::Code>> {};

class PointScalarTest
  : public PrimitiveScalarUnit,
    public ::testing::WithParamInterface<std::tuple<std::vector<legate::coord_t>, std::int32_t>> {};

class RectScalarTest
  : public PrimitiveScalarUnit,
    public ::testing::WithParamInterface<
      std::tuple<std::vector<legate::coord_t>, std::vector<legate::coord_t>, std::int32_t>> {};

// NOLINTBEGIN(readability-magic-numbers)
std::vector<std::tuple<std::vector<legate::coord_t>, std::int32_t>> scalar_with_point_cases()
{
  std::vector<std::tuple<std::vector<legate::coord_t>, std::int32_t>> cases = {
    {{0}, 1}, {{1, 8}, 2}, {{-10, 2, -1}, 3}, {{1, 5, 7, 200}, 4}};

#if LEGATE_MAX_DIM >= 5
  cases.emplace_back(
    std::tuple<std::vector<legate::coord_t>, std::int32_t>({{10, 10, 10, 10, 10}, 5}));
#endif
#if LEGATE_MAX_DIM >= 6
  cases.emplace_back(
    std::tuple<std::vector<legate::coord_t>, std::int32_t>({{-1, 0, 5, -10, 90, 1}, 6}));
#endif
#if LEGATE_MAX_DIM >= 7
  cases.emplace_back(
    std::tuple<std::vector<legate::coord_t>, std::int32_t>({{2, 3, 4, 5, 6, 7, 8}, 7}));
#endif

  return cases;
}

std::vector<std::tuple<std::vector<legate::coord_t>, std::vector<legate::coord_t>, std::int32_t>>
scalar_with_rect_cases()
{
  std::vector<std::tuple<std::vector<legate::coord_t>, std::vector<legate::coord_t>, std::int32_t>>
    cases = {{{1}, {-9}, 1},
             {{-1, 3}, {9, -2}, 2},
             {{0, 1, 2}, {3, 4, 5}, 3},
             {{-5, 1, -7, 10}, {4, 5, 6, 7}, 4}};

#if LEGATE_MAX_DIM >= 5
  cases.emplace_back(
    std::tuple<std::vector<legate::coord_t>, std::vector<legate::coord_t>, std::int32_t>(
      {{1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}, 5}));
#endif
#if LEGATE_MAX_DIM >= 6
  cases.emplace_back(
    std::tuple<std::vector<legate::coord_t>, std::vector<legate::coord_t>, std::int32_t>(
      {{-10, 2, 50, 0, 2, 1}, {7, 10, 5, 8, -1, 1}, 6}));
#endif
#if LEGATE_MAX_DIM >= 7
  cases.emplace_back(
    std::tuple<std::vector<legate::coord_t>, std::vector<legate::coord_t>, std::int32_t>(
      {{0, 0, 0, 0, 0, 0, 0}, {1, 4, 1, 7, 2, 5, -1}, 7}));
#endif

  return cases;
}
// NOLINTEND(readability-magic-numbers)

INSTANTIATE_TEST_SUITE_P(
  PrimitiveScalarUnit,
  PrimitiveScalarTest,
  ::testing::Values(
    std::make_tuple(legate::Scalar{BOOL_VALUE}, legate::Type::Code::BOOL),
    std::make_tuple(legate::Scalar{BOOL_VALUE, legate::bool_()}, legate::Type::Code::BOOL),
    std::make_tuple(legate::Scalar{INT8_VALUE}, legate::Type::Code::INT8),
    std::make_tuple(legate::Scalar{INT8_VALUE, legate::int8()}, legate::Type::Code::INT8),
    std::make_tuple(legate::Scalar{INT16_VALUE}, legate::Type::Code::INT16),
    std::make_tuple(legate::Scalar{INT16_VALUE, legate::int16()}, legate::Type::Code::INT16),
    std::make_tuple(legate::Scalar{INT32_VALUE}, legate::Type::Code::INT32),
    std::make_tuple(legate::Scalar{INT32_VALUE, legate::int32()}, legate::Type::Code::INT32),
    std::make_tuple(legate::Scalar{INT64_VALUE}, legate::Type::Code::INT64),
    std::make_tuple(legate::Scalar{INT64_VALUE, legate::int64()}, legate::Type::Code::INT64),
    std::make_tuple(legate::Scalar{UINT8_VALUE}, legate::Type::Code::UINT8),
    std::make_tuple(legate::Scalar{UINT8_VALUE, legate::uint8()}, legate::Type::Code::UINT8),
    std::make_tuple(legate::Scalar{UINT16_VALUE}, legate::Type::Code::UINT16),
    std::make_tuple(legate::Scalar{UINT16_VALUE, legate::uint16()}, legate::Type::Code::UINT16),
    std::make_tuple(legate::Scalar{UINT32_VALUE}, legate::Type::Code::UINT32),
    std::make_tuple(legate::Scalar{UINT32_VALUE, legate::uint32()}, legate::Type::Code::UINT32),
    std::make_tuple(legate::Scalar{UINT64_VALUE}, legate::Type::Code::UINT64),
    std::make_tuple(legate::Scalar{UINT64_VALUE, legate::uint64()}, legate::Type::Code::UINT64),
    std::make_tuple(legate::Scalar{FLOAT16_VALUE}, legate::Type::Code::FLOAT16),
    std::make_tuple(legate::Scalar{FLOAT16_VALUE, legate::float16()}, legate::Type::Code::FLOAT16),
    std::make_tuple(legate::Scalar{FLOAT_VALUE}, legate::Type::Code::FLOAT32),
    std::make_tuple(legate::Scalar{FLOAT_VALUE, legate::float32()}, legate::Type::Code::FLOAT32),
    std::make_tuple(legate::Scalar{DOUBLE_VALUE}, legate::Type::Code::FLOAT64),
    std::make_tuple(legate::Scalar{DOUBLE_VALUE, legate::float64()}, legate::Type::Code::FLOAT64),
    std::make_tuple(legate::Scalar{COMPLEX_FLOAT_VALUE}, legate::Type::Code::COMPLEX64),
    std::make_tuple(legate::Scalar{COMPLEX_FLOAT_VALUE, legate::complex64()},
                    legate::Type::Code::COMPLEX64),
    std::make_tuple(legate::Scalar{COMPLEX_DOUBLE_VALUE}, legate::Type::Code::COMPLEX128),
    std::make_tuple(legate::Scalar{COMPLEX_DOUBLE_VALUE, legate::complex128()},
                    legate::Type::Code::COMPLEX128),
    std::make_tuple(legate::Scalar{SmallEnumType::FOO}, legate::Type::Code::UINT8),
    std::make_tuple(legate::Scalar{SmallEnumType::FOO, legate::uint8()}, legate::Type::Code::UINT8),
    std::make_tuple(legate::Scalar{BigEnumType::BAR}, legate::Type::Code::INT64),
    std::make_tuple(legate::Scalar{BigEnumType::BAR, legate::int64()}, legate::Type::Code::INT64)));

INSTANTIATE_TEST_SUITE_P(PrimitiveScalarUnit,
                         PointScalarTest,
                         ::testing::ValuesIn(scalar_with_point_cases()));

INSTANTIATE_TEST_SUITE_P(PrimitiveScalarUnit,
                         RectScalarTest,
                         ::testing::ValuesIn(scalar_with_rect_cases()));

template <std::int32_t DIM>
void check_rect_bounds(const MultiDimRectStruct<DIM>& to_compare,
                       const MultiDimRectStruct<DIM>& expect)
{
  ASSERT_EQ(std::size(to_compare.lo), std::size(to_compare.hi));
  ASSERT_EQ(std::size(expect.lo), std::size(expect.hi));
  ASSERT_EQ(std::size(to_compare.lo), std::size(expect.lo));

  for (std::size_t i = 0; i < std::size(to_compare.lo); i++) {
    ASSERT_EQ(to_compare.lo[i], expect.lo[i]);
    ASSERT_EQ(to_compare.hi[i], expect.hi[i]);
  }
}

template <typename T>
void check_string_scalar_values()
{
  const legate::Scalar scalar{STRING_VALUE};
  const auto values        = scalar.values<T>();
  const auto actual_values = legate::Span<const T>{values.data(), values.size()};
  const auto actual_value  = scalar.value<std::string>();

  ASSERT_EQ(actual_values.size(), actual_value.size());
  ASSERT_EQ(*actual_values.begin(), actual_value[0]);
}

class CheckPrimitiveScalarFn {
 public:
  template <legate::Type::Code CODE>
  void operator()(const legate::Scalar& scalar, legate::Type::Code code) const
  {
    using T = legate::type_of_t<CODE>;

    ASSERT_EQ(scalar.type().code(), code);
    ASSERT_EQ(scalar.size(), sizeof(T));
    ASSERT_EQ(scalar.values<T>().size(), 1);
    ASSERT_NE(scalar.ptr(), nullptr);
  }
};

class CheckPointScalarFn {
 public:
  template <std::int32_t DIM>
  void operator()(const std::vector<legate::coord_t>& bounds) const
  {
    const auto point = legate::Point<DIM>{bounds.data()};
    const legate::Scalar scalar{point};
    auto fixed_type = legate::fixed_array_type(legate::int64(), DIM);

    ASSERT_EQ(scalar.type().code(), legate::Type::Code::FIXED_ARRAY);
    ASSERT_EQ(scalar.size(), fixed_type.size());
    ASSERT_NE(scalar.ptr(), nullptr);

    // Check values
    const auto expected_values = legate::Span<const legate::coord_t>{bounds};
    const auto actual_values =
      legate::Span<const legate::coord_t>{scalar.values<legate::coord_t>()};

    ASSERT_EQ(actual_values.size(), DIM);
    ASSERT_EQ(actual_values.size(), expected_values.size());
    for (int i = 0; i < DIM; i++) {
      ASSERT_EQ(actual_values[i], expected_values[i]);
    }
  }
};

class CheckRectScalarFn {
 public:
  template <std::int32_t DIM>
  void operator()(const std::vector<legate::coord_t>& lo,
                  const std::vector<legate::coord_t>& hi) const
  {
    auto point_lo = legate::Point<DIM>{lo.data()};
    auto point_hi = legate::Point<DIM>{hi.data()};
    auto rect     = legate::Rect<DIM>{point_lo, point_hi};
    const legate::Scalar scalar{rect};
    auto struct_type = legate::struct_type(/* align */ true,
                                           legate::fixed_array_type(legate::int64(), DIM),
                                           legate::fixed_array_type(legate::int64(), DIM));

    ASSERT_EQ(scalar.type().code(), legate::Type::Code::STRUCT);
    ASSERT_EQ(scalar.size(), struct_type.size());
    ASSERT_NE(scalar.ptr(), nullptr);

    // Check values
    auto actual_data = scalar.value<MultiDimRectStruct<DIM>>();
    MultiDimRectStruct<DIM> expected_data;

    for (int i = 0; i < DIM; i++) {
      expected_data.lo[i] = lo[i];
      expected_data.hi[i] = hi[i];
    }
    check_rect_bounds(actual_data, expected_data);

    const auto actual_values =
      legate::Span<const MultiDimRectStruct<DIM>>{scalar.values<MultiDimRectStruct<DIM>>()};
    const auto expected_values = legate::Span<const MultiDimRectStruct<DIM>>{&expected_data, 1};

    ASSERT_EQ(actual_values.size(), 1);
    ASSERT_EQ(actual_values.size(), expected_values.size());
    ASSERT_NE(actual_values.ptr(), expected_values.ptr());
    check_rect_bounds(*actual_values.begin(), *expected_values.begin());
  }
};

}  // namespace

TEST_F(PrimitiveScalarUnit, Object)
{
  // constructor with Scalar object
  const legate::Scalar scalar1{INT32_VALUE};
  const legate::Scalar scalar2{scalar1};  // NOLINT(performance-unnecessary-copy-initialization)

  ASSERT_EQ(scalar2.type().code(), scalar1.type().code());
  ASSERT_EQ(scalar2.size(), scalar1.size());
  ASSERT_EQ(scalar2.size(), legate::uint32().size());
  ASSERT_NE(scalar2.ptr(), nullptr);
  ASSERT_EQ(scalar2.value<std::int32_t>(), scalar1.value<std::int32_t>());
  ASSERT_EQ(scalar2.values<std::int32_t>().size(), scalar1.values<std::int32_t>().size());
  ASSERT_EQ(*scalar2.values<std::int32_t>().begin(), *scalar1.values<std::int32_t>().begin());
}

TEST_F(PrimitiveScalarUnit, SharedScalar)
{
  const auto data_vec = std::vector<std::uint64_t>(DATA_SIZE, UINT64_VALUE);
  const auto* data    = data_vec.data();
  const legate::Scalar scalar{legate::uint64(), data};

  ASSERT_EQ(scalar.type().code(), legate::Type::Code::UINT64);
  ASSERT_EQ(scalar.size(), legate::uint64().size());
  ASSERT_EQ(scalar.ptr(), data);
  ASSERT_EQ(scalar.value<std::uint64_t>(), UINT64_VALUE);

  const auto actual_values   = legate::Span<const std::uint64_t>{scalar.values<std::uint64_t>()};
  const auto expected_values = legate::Span<const std::uint64_t>{data, 1};

  ASSERT_EQ(*actual_values.begin(), *expected_values.begin());
  ASSERT_EQ(actual_values.size(), expected_values.size());
}

TEST_F(PrimitiveScalarUnit, OwnedSharedScalar)
{
  const auto data_vec = std::vector<std::uint32_t>(DATA_SIZE, UINT32_VALUE);
  const auto* data    = data_vec.data();
  const legate::Scalar scalar{legate::uint32(), data, /* copy */ true};

  ASSERT_NE(scalar.ptr(), data);
}

TEST_P(PrimitiveScalarTest, Create)
{
  const auto [scalar, code] = GetParam();

  legate::type_dispatch(code, CheckPrimitiveScalarFn{}, scalar, code);
}

TEST_F(PrimitiveScalarUnit, String)
{
  // constructor with string
  auto input_string = STRING_VALUE;
  const legate::Scalar scalar{input_string};

  ASSERT_EQ(scalar.type().code(), legate::Type::Code::STRING);

  auto expected_size = sizeof(std::uint32_t) + (sizeof(char) * input_string.size());

  ASSERT_EQ(scalar.size(), expected_size);
  ASSERT_NE(scalar.ptr(), input_string.data());

  // Check values
  ASSERT_EQ(scalar.value<std::string>(), input_string);
  ASSERT_EQ(scalar.value<std::string_view>(), input_string);

  const auto actual_values = legate::Span<const char>{scalar.values<char>()};

  ASSERT_EQ(actual_values.size(), input_string.size());
  ASSERT_EQ(*actual_values.begin(), input_string[0]);
}

TEST_P(PointScalarTest, Create)
{
  const auto [bounds, DIM] = GetParam();

  legate::dim_dispatch(DIM, CheckPointScalarFn{}, bounds);
}

TEST_P(RectScalarTest, Create)
{
  const auto [lo, hi, DIM] = GetParam();

  legate::dim_dispatch(DIM, CheckRectScalarFn{}, lo, hi);
}

TEST_F(PrimitiveScalarUnit, Empty)
{
  const legate::Scalar scalar{};

  ASSERT_EQ(scalar.type().code(), legate::Type::Code::NIL);
  ASSERT_EQ(scalar.size(), 0);
  ASSERT_EQ(scalar.ptr(), nullptr);
  ASSERT_THROW(static_cast<void>(scalar.value<std::int64_t>()), std::invalid_argument);

  const auto actual_values = legate::Span<const std::int64_t>{scalar.values<std::int64_t>()};

  ASSERT_EQ(actual_values.begin(), nullptr);
  ASSERT_EQ(actual_values.size(), 0);
}

TEST_F(PrimitiveScalarUnit, Null)
{
  const legate::Scalar null_scalar = legate::null();

  ASSERT_EQ(null_scalar.type().code(), legate::Type::Code::NIL);
  ASSERT_EQ(null_scalar.size(), 0);
  ASSERT_EQ(null_scalar.ptr(), nullptr);
  ASSERT_THROW(static_cast<void>(null_scalar.value<std::int32_t>()), std::invalid_argument);

  const legate::Span actual_values{null_scalar.values<std::int64_t>()};

  ASSERT_EQ(actual_values.begin(), nullptr);
  ASSERT_EQ(actual_values.size(), 0);
}

TEST_F(PrimitiveScalarUnit, OperatorEqual)
{
  const legate::Scalar scalar1{INT32_VALUE};
  legate::Scalar scalar2{UINT64_VALUE};

  scalar2 = scalar1;
  ASSERT_EQ(scalar2.type().code(), scalar1.type().code());
  ASSERT_EQ(scalar2.size(), scalar1.size());
  ASSERT_EQ(scalar2.value<std::int32_t>(), scalar1.value<std::int32_t>());
  ASSERT_EQ(scalar2.values<std::int32_t>().size(), scalar1.values<std::int32_t>().size());
}

TEST_F(PrimitiveScalarUnit, StringScalarValues)
{
  check_string_scalar_values<char>();
  check_string_scalar_values<std::int8_t>();
  check_string_scalar_values<std::uint8_t>();
}

}  // namespace create_primitive_scalar_test
