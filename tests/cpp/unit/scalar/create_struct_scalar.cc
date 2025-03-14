/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/scalar.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace create_struct_scalar_test {

namespace {

using StructScalarUnit = DefaultFixture;

constexpr bool BOOL_VALUE            = true;
constexpr std::int32_t INT32_VALUE   = 2700;
constexpr std::uint64_t UINT64_VALUE = 100;

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

template <typename T>
void check_binary_scalar(T& value)
{
  const legate::Scalar scalar{value, legate::binary_type(sizeof(T))};

  ASSERT_EQ(scalar.type().size(), sizeof(T));
  ASSERT_NE(scalar.ptr(), nullptr);
  ASSERT_EQ(value, scalar.value<T>());

  auto actual = scalar.values<T>();

  ASSERT_EQ(actual.size(), 1);
  ASSERT_EQ(actual[0], value);
}

template <typename T>
void check_struct_type_scalar(T& struct_data, bool align)
{
  const legate::Scalar scalar{
    struct_data, legate::struct_type(align, legate::bool_(), legate::int32(), legate::uint64())};

  ASSERT_EQ(scalar.type().code(), legate::Type::Code::STRUCT);
  ASSERT_EQ(scalar.size(), sizeof(T));
  ASSERT_NE(scalar.ptr(), nullptr);

  // Check value
  T actual_data = scalar.value<T>();
  // When taking the address (or reference!) of a packed struct member, the resulting
  // pointer/reference is _not_ aligned to its normal type. This is a problem when other
  // functions (like ASSERT_EQ()) take their arguments by reference.
  //
  // So we need to make copies of the values (which will be properly aligned since they are
  // locals), in order for this not to raise UBSAN errors.
  auto compare = [](auto lhs, auto rhs) { ASSERT_EQ(lhs, rhs); };

  compare(actual_data.bool_data, struct_data.bool_data);
  compare(actual_data.int32_data, struct_data.int32_data);
  compare(actual_data.uint64_data, struct_data.uint64_data);

  // Check values
  const auto values          = scalar.values<T>();
  const auto actual_values   = legate::Span<const T>{values.data(), values.size()};
  const auto expected_values = legate::Span<const T>{&struct_data, 1};

  ASSERT_EQ(actual_values.size(), expected_values.size());
  ASSERT_NE(actual_values.ptr(), expected_values.ptr());
  compare(actual_values.begin()->bool_data, expected_values.begin()->bool_data);
  compare(actual_values.begin()->int32_data, expected_values.begin()->int32_data);
  compare(actual_values.begin()->uint64_data, expected_values.begin()->uint64_data);
}

}  // namespace

TEST_F(StructScalarUnit, CreateWithBinaryTypePadding)
{
  PaddingStructData value = {BOOL_VALUE, INT32_VALUE, UINT64_VALUE};

  check_binary_scalar(value);
}

TEST_F(StructScalarUnit, CreateWithBinaryTypeNoPadding)
{
  NoPaddingStructData value = {BOOL_VALUE, INT32_VALUE, UINT64_VALUE};

  check_binary_scalar(value);
}

TEST_F(StructScalarUnit, CreateWithStructTypePadding)
{
  PaddingStructData struct_data = {BOOL_VALUE, INT32_VALUE, UINT64_VALUE};

  check_struct_type_scalar(struct_data, /* align*/ true);
}

TEST_F(StructScalarUnit, CreateWithStructTypeNoPadding)
{
  NoPaddingStructData struct_data = {BOOL_VALUE, INT32_VALUE, UINT64_VALUE};

  check_struct_type_scalar(struct_data, /* align*/ false);
}

}  // namespace create_struct_scalar_test
