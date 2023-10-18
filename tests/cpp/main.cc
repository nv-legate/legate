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

#include <gtest/gtest.h>
#include "legate.h"
#include "utilities/utilities.h"

extern "C" {

const char* __asan_default_options()  // NOLINT(bugprone-reserved-identifier)
{
  return "check_initialization_order=1:"
         "detect_stack_use_after_return=1:"
         "alloc_dealloc_mismatch=1:"
         "strict_string_checks=1:"
         "color=always:"
         // note trailing ":", this is so that user may write ASAN_OPTIONS+="foo:bar:baz"
         "symbolize=1:";
}

const char* __lsan_default_suppressions()  // NOLINT(bugprone-reserved-identifier)
{
  return "leak:librealm.*\n"
         "leak:liblegion.*\n";
}
//
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  DefaultFixture::init(argc, argv);
  DeathTestFixture::init(argc, argv);

  return RUN_ALL_TESTS();
}
