/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <unit/logical_array/utils.h>

#include <legate.h>

namespace logical_array_util_test {

legate::LogicalArray create_array_with_type(const legate::Type& type,
                                            bool bound,
                                            bool nullable,
                                            bool optimize_scalar)
{
  auto runtime               = legate::Runtime::get_runtime();
  constexpr auto UNBOUND_DIM = 1;

  if (bound) {
    const auto shape = type.variable_size() ? legate::Shape{10} : legate::Shape{1, 2, 3, 4};
    return runtime->create_array(shape, type, nullable, optimize_scalar);
  }

  return runtime->create_array(type, UNBOUND_DIM, nullable);
}

const legate::StructType& struct_type()
{
  static const auto t =
    legate::struct_type(true, legate::uint16(), legate::int64(), legate::float32())
      .as_struct_type();

  return t;
}

const legate::ListType& list_type()
{
  static const auto t = legate::list_type(legate::int64()).as_list_type();

  return t;
}

}  // namespace logical_array_util_test
