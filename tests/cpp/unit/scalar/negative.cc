/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/scalar.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace negative_scalar_test {

namespace {

using NegativeScalarUnit = DefaultFixture;

}  // namespace

TEST_F(NegativeScalarUnit, SizeMismatch)
{
  constexpr double DOUBLE_VALUE = 4.6;

  ASSERT_THROW((legate::Scalar{DOUBLE_VALUE, legate::float32()}), std::invalid_argument);
}

TEST_F(NegativeScalarUnit, InvalidType)
{
  ASSERT_THROW((legate::Scalar{1, legate::null_type()}), std::invalid_argument);
}

TEST_F(NegativeScalarUnit, InvalidValuePrimitiveScalar)
{
  const legate::Scalar scalar{1, legate::uint32()};

  ASSERT_THROW(static_cast<void>(scalar.value<std::int64_t>()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(scalar.values<std::string>()), std::invalid_argument);
}

TEST_F(NegativeScalarUnit, InvalidValueStringScalar)
{
  const legate::Scalar scalar{"hello"};

  ASSERT_THROW(static_cast<void>(scalar.value<std::int64_t>()), std::invalid_argument);
}

TEST_F(NegativeScalarUnit, InvalidValueFixedArrayScalar)
{
  const std::vector<std::int32_t> scalar_data{1, 2};
  const legate::Scalar scalar{scalar_data};

  ASSERT_THROW(static_cast<void>(scalar.value<std::int32_t>()), std::invalid_argument);
}

TEST_F(NegativeScalarUnit, InvalidValuesPrimitiveScalar)
{
  const legate::Scalar scalar{1, legate::uint32()};

  ASSERT_THROW(static_cast<void>(scalar.values<std::int64_t>()), std::invalid_argument);
}

TEST_F(NegativeScalarUnit, InvalidValuesStringScalar)
{
  const legate::Scalar scalar{"world"};

  ASSERT_THROW(static_cast<void>(scalar.values<bool>()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(scalar.values<std::int16_t>()), std::invalid_argument);
}

TEST_F(NegativeScalarUnit, InvalidValuesFixedArrayScalar)
{
  const std::vector<std::int32_t> scalar_data{1, 2};
  const legate::Scalar scalar{scalar_data};

  ASSERT_THROW(static_cast<void>(scalar.values<std::int64_t>()), std::invalid_argument);
}

}  // namespace negative_scalar_test
