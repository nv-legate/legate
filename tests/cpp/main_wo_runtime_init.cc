/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate_defines.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace {

class ThrowListener : public ::testing::EmptyTestEventListener {
 public:
  void OnTestPartResult(const ::testing::TestPartResult& result) override
  {
    if (result.type() == ::testing::TestPartResult::kFatalFailure) {
      throw testing::AssertionException{result};
    }
  }
};

}  // namespace

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  GTEST_FLAG_SET(death_test_style, "threadsafe");

  ::testing::UnitTest::GetInstance()->listeners().Append(new ThrowListener);

  return RUN_ALL_TESTS();
}
