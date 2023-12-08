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

#include "legate_defines.h"
#include "utilities/utilities.h"

#include <gtest/gtest.h>

extern "C" {

// NOLINTBEGIN(bugprone-reserved-identifier)
const char* __asan_default_options()
{
  return "check_initialization_order=1:"
         "detect_stack_use_after_return=1:"
         "alloc_dealloc_mismatch=1:"
         "strict_string_checks=1:"
         "color=always:"
         "detect_odr_violation=2:"
#if LegateDefined(LEGATE_USE_CUDA)
         "protect_shadow_gap=0:"
#endif
         // note trailing ":", this is so that user may write ASAN_OPTIONS+="foo:bar:baz"
         "symbolize=1:";
}

const char* __ubsan_default_options() { return "print_stacktrace=1:"; }

const char* __lsan_default_suppressions()
{
  return "leak:librealm.*\n"
         "leak:liblegion.*\n";
}
// NOLINTEND(bugprone-reserved-identifier)
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  DefaultFixture::init(argc, argv);
  DeathTestFixture::init(argc, argv);

  return RUN_ALL_TESTS();
}
