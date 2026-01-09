/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/experimental/stl/detail/registrar.hpp>

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

  try {
    legate::start();
  } catch (const std::exception& e) {
    [&] { FAIL() << "Legate failed to start: " << e.what(); }();
    return 1;
  }

  try {
    const legate::experimental::stl::initialize_library init{};
  } catch (const std::exception& exn) {
    [&] { FAIL() << "Legate STL failed to start: " << exn.what(); }();
    return 1;
  }

  ::testing::UnitTest::GetInstance()->listeners().Append(new ThrowListener);

  auto result = RUN_ALL_TESTS();

  if (result) {
    // handle error from RUN_ALL_TESTS()
    return result;
  }

  result = legate::finish();
  EXPECT_EQ(result, 0);

  return result;
}
