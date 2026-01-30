/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/scalar.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace negative_scalar_test {

namespace {

using NegativeScalarUnit = DefaultFixture;

}  // namespace

TEST_F(NegativeScalarUnit, SizeMismatch)
{
  constexpr double DOUBLE_VALUE = 4.6;

  ASSERT_THAT([&] { static_cast<void>(legate::Scalar{DOUBLE_VALUE, legate::float32()}); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Size of the value doesn't match with the type")));
}

TEST_F(NegativeScalarUnit, InvalidType)
{
  ASSERT_THAT([&] { static_cast<void>(legate::Scalar{1, legate::null_type()}); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Null type cannot be used")));
}

TEST_F(NegativeScalarUnit, InvalidValuePrimitiveScalar)
{
  const legate::Scalar scalar{1, legate::uint32()};

  ASSERT_THAT([&] { static_cast<void>(scalar.value<std::int64_t>()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("requested type has size")));
  ASSERT_THAT([&] { static_cast<void>(scalar.values<std::string>()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("requested type has size")));
}

TEST_F(NegativeScalarUnit, InvalidValueStringScalar)
{
  const legate::Scalar scalar{"hello"};

  ASSERT_THAT([&] { static_cast<void>(scalar.value<std::int64_t>()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("string cannot be casted to other types")));
}

TEST_F(NegativeScalarUnit, InvalidStringViewType)
{
  const legate::Scalar scalar{std::int32_t{1}};

  ASSERT_THAT([&] { static_cast<void>(scalar.value<std::string_view>()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Type of the scalar is not string")));
}

TEST_F(NegativeScalarUnit, InvalidValueFixedArrayScalar)
{
  const std::vector<std::int32_t> scalar_data{1, 2};
  const legate::Scalar scalar{scalar_data};

  ASSERT_THAT([&] { static_cast<void>(scalar.value<std::int32_t>()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("requested type has size")));
}

TEST_F(NegativeScalarUnit, InvalidValuesPrimitiveScalar)
{
  const legate::Scalar scalar{1, legate::uint32()};

  ASSERT_THAT([&] { static_cast<void>(scalar.values<std::int64_t>()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("requested type has size")));
}

TEST_F(NegativeScalarUnit, InvalidValuesStringScalar)
{
  const legate::Scalar scalar{"world"};

  ASSERT_THAT([&] { static_cast<void>(scalar.values<bool>()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("string cannot be casted to Span<bool>")));
  ASSERT_THAT([&] { static_cast<void>(scalar.values<std::int16_t>()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("scalar can only be converted into a span")));
}

TEST_F(NegativeScalarUnit, InvalidValuesFixedArrayScalar)
{
  const std::vector<std::int32_t> scalar_data{1, 2};
  const legate::Scalar scalar{scalar_data};

  ASSERT_THAT([&] { static_cast<void>(scalar.values<std::int64_t>()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("requested type has size")));
}

}  // namespace negative_scalar_test
