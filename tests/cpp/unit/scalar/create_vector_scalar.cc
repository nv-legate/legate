/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/scalar.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace create_vector_scalar_test {

namespace {

using VectorScalarUnit = DefaultFixture;

}  // namespace

TEST_F(VectorScalarUnit, CreateWithVector)
{
  // constructor with arrays
  constexpr std::int32_t INT32_VALUE = 200;
  const std::vector<std::int32_t> scalar_data{INT32_VALUE, INT32_VALUE};
  const legate::Scalar scalar{scalar_data};
  auto fixed_type = legate::fixed_array_type(legate::int32(), scalar_data.size());

  ASSERT_EQ(scalar.type().code(), legate::Type::Code::FIXED_ARRAY);
  ASSERT_EQ(scalar.size(), fixed_type.size());

  const std::vector<std::int32_t> data_vec = {INT32_VALUE, INT32_VALUE};
  const auto* data                         = data_vec.data();
  const auto expected_values = legate::Span<const std::int32_t>{data, scalar_data.size()};
  const auto actual_values   = legate::Span<const std::int32_t>{scalar.values<std::int32_t>()};

  ASSERT_EQ(actual_values.size(), expected_values.size());
  for (std::size_t i = 0; i < scalar_data.size(); i++) {
    ASSERT_EQ(actual_values[i], expected_values[i]);
  }
}

TEST_F(VectorScalarUnit, CreateWithEmptyVector)
{
  const auto vec    = std::vector<std::uint32_t>{};
  const auto scalar = legate::Scalar{vec};  // The construction of this should not throw

  ASSERT_EQ(scalar.size(), 0);
}

TEST_F(VectorScalarUnit, CreateWithVectorBool)
{
  constexpr auto SIZE = 13;
  auto vec            = std::vector<bool>{};

  std::generate_n(std::back_inserter(vec), SIZE, [i = 0]() mutable -> bool { return i++ % 2; });

  const auto scal = legate::Scalar{vec};

  ASSERT_EQ(scal.type(), legate::fixed_array_type(legate::bool_(), vec.size()));
  ASSERT_EQ(scal.size(), vec.size());

  const auto values = scal.values<bool>();

  ASSERT_EQ(values.size(), vec.size());
  for (std::size_t i = 0; i < vec.size(); ++i) {
    ASSERT_EQ(values[i], vec[i]);
  }
}

}  // namespace create_vector_scalar_test
