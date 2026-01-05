/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace task_exception_test {

using TaskException = DefaultFixture;

TEST_F(TaskException, Basic)
{
  constexpr const char* ERROR_MSG1 = "Exception Test1";

  const legate::TaskException exc1{ERROR_MSG1};
  EXPECT_STREQ(exc1.what(), ERROR_MSG1);

  constexpr const char* ERROR_MSG2 = "Exception Test2";
  const legate::TaskException exc2{ERROR_MSG2};

  EXPECT_STREQ(exc2.what(), ERROR_MSG2);
}

}  // namespace task_exception_test
