/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace test_offload_to {

class OffloadToUnit : public DefaultFixture {};

TEST_F(OffloadToUnit, NullableArray)
{
  auto* runtime = legate::Runtime::get_runtime();
  auto array = runtime->create_array(legate::Shape{3, 3, 3}, legate::int32(), /* nullable */ true);

  // Silence any Legion warnings about uninitialized data
  runtime->issue_fill(array, legate::Scalar{std::int32_t{1}});
  // This should not throw an exception about mismatched signature for the offload task due to
  // invalid signature. See https://github.com/nv-legate/legate.internal/issues/1930
  ASSERT_NO_THROW(array.offload_to(legate::mapping::StoreTarget::SYSMEM));
}

}  // namespace test_offload_to
