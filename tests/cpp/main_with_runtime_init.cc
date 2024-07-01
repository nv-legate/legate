/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "core/experimental/stl/detail/registrar.hpp"
#include "legate.h"
#include "utilities/sanitizer_options.h"
#include "utilities/utilities.h"

#include <gtest/gtest.h>

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  GTEST_FLAG_SET(death_test_style, "fast");

  if (auto result = legate::start(argc, argv); result != 0) {
    [&result] { FAIL() << "Legate failed to start: " << result; }();
    return result;
  }

  try {
    const legate::experimental::stl::initialize_library init{};
  } catch (const std::exception& exn) {
    std::cerr << exn.what() << '\n';
  }

  auto result = RUN_ALL_TESTS();

  if (result) {
    // handle error from RUN_ALL_TESTS()
    return result;
  }

  result = legate::finish();
  EXPECT_EQ(result, 0);

  return result;
}
