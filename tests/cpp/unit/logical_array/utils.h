/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace logical_array_util_test {

[[nodiscard]] legate::LogicalArray create_array_with_type(const legate::Type& type,
                                                          bool bound,
                                                          bool nullable,
                                                          bool optimize_scalar = false);

[[nodiscard]] const legate::StructType& struct_type();

[[nodiscard]] const legate::ListType& list_type();

constexpr auto BOUND_DIM                = 4;
constexpr auto VARILABLE_TYPE_BOUND_DIM = 1;
constexpr auto UNBOUND_DIM              = 1;
constexpr auto NUM_CHILDREN             = 2;

inline const legate::Shape& bound_shape_multi_dim()
{
  static const auto shape = legate::Shape{1, 2, 3, 4};
  return shape;
}

inline const legate::Shape& bound_shape_single_dim()
{
  static const auto shape = legate::Shape{10};
  return shape;
}

}  // namespace logical_array_util_test
