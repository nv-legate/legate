/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/runtime/detail/runtime.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace runtime_exception_test {

using RuntimeNoInit = ::testing::Test;

TEST_F(RuntimeNoInit, GetRuntime)
{
  ASSERT_NO_THROW(legate::start());
  ASSERT_EQ(legate::finish(), 0);
  EXPECT_THROW(static_cast<void>(legate::Runtime::get_runtime()), std::runtime_error);
}

}  // namespace runtime_exception_test
