/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */
#include <legate.h>

#include <legate/utilities/detail/strtoll.h>

#include <gtest/gtest.h>

#include <cstdlib>
#include <unistd.h>
#include <utilities/utilities.h>

namespace death_example {

using ExampleDeathTest = DefaultFixture;

namespace {

void kill_process()
{
  legate::start();
  std::abort();
}

}  // namespace

// __has_feature(address_sanitizer) is available only on Clang, so we unify it to
// __SANITIZE_ADDRESS__ that GCC sets
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define __SANITIZE_ADDRESS__
#endif
#endif

// FIXME(wonchanl): this test hangs on aarch64 when the sanitizer is enabled. Since issues like this
// can easily take really long to figure out and this test doesn't benefit from the sanitizer, we
// disable it for now.
#if LEGATE_DEFINED(__aarch64__) && LEGATE_DEFINED(__SANITIZE_ADDRESS__)
TEST_F(ExampleDeathTest, DISABLED_Simple)
#else
TEST_F(ExampleDeathTest, Simple)
#endif
{
  const auto value           = std::getenv("REALM_BACKTRACE");
  const bool realm_backtrace = value != nullptr && legate::detail::safe_strtoll(value) != 0;

  if (realm_backtrace) {
    // We can't check that the subprocess dies with SIGABRT when we run with REALM_BACKTRACE=1,
    // because Realm's signal handler doesn't propagate the signal, instead it does exit(1).
    // Even worse, when ASAN is used this triggers a segfault in Realm's signal handler, which
    // causes it to abort() instead of exit(1), so for now we just don't check the exit code
    // at all.

    EXPECT_DEATH(kill_process(), "");
  } else {
    EXPECT_EXIT(kill_process(), ::testing::KilledBySignal(SIGABRT), "");
  }
}

}  // namespace death_example
