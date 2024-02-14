/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "legate.h"
#include "utilities/utilities.h"

#include <gtest/gtest.h>

namespace task_exception_test {

using TaskException = DefaultFixture;

TEST_F(TaskException, Basic)
{
  std::int32_t index1    = 100;
  const char* ERROR_MSG1 = "Exception Test1";
  legate::TaskException exc1{index1, ERROR_MSG1};
  EXPECT_EQ(exc1.index(), index1);
  EXPECT_STREQ(exc1.error_message().c_str(), ERROR_MSG1);

  const char* ERROR_MSG2 = "Exception Test2";
  legate::TaskException exc2{ERROR_MSG2};
  EXPECT_EQ(exc2.index(), 0);
  EXPECT_STREQ(exc2.error_message().c_str(), ERROR_MSG2);
}

}  // namespace task_exception_test
