/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/scope_guard.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace runtime_init_test {

class RuntimeInitUnit : public ::testing::Test {};

TEST_F(RuntimeInitUnit, GetRuntime)
{
  ASSERT_THAT([&] { static_cast<void>(legate::detail::Runtime::get_runtime()); },
              ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
                "Must call legate::start() before retrieving the Legate runtime.")));
  ASSERT_NO_THROW(legate::start());
  ASSERT_EQ(legate::finish(), 0);
  ASSERT_TRUE(legate::has_finished());
  ASSERT_FALSE(legate::has_started());
  ASSERT_THAT(
    [&] { static_cast<void>(legate::Runtime::get_runtime()); },
    ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
      "Legate runtime has not been initialized. Please invoke legate::start to use the runtime")));
  ASSERT_THAT([&] { static_cast<void>(legate::detail::Runtime::get_runtime()); },
              ::testing::ThrowsMessage<std::runtime_error>(
                ::testing::HasSubstr("Legate runtime has been finalized, and cannot be "
                                     "re-initialized without restarting the program.")));
  ASSERT_TRUE(legate::has_finished());
  ASSERT_FALSE(legate::has_started());
}

// If we have ASAN (or, specifically, if we have *LSAN*) then this test has a bunch of spurious
// memory leaks, likely stemming from Legion not shutting down UCX properly. It's not clear
// where they are coming from, LSAN only reports:
//
// ==1765==ERROR: LeakSanitizer: detected memory leaks
//
// Direct leak of 4816 byte(s) in 7 object(s) allocated from:
//    #0 0x7f29f0f7807b in __interceptor_malloc libsanitizer/asan/asan_malloc_linux.cpp:145
//    #1 0x7f2944853337  (<unknown module>)
//
// Direct leak of 4816 byte(s) in 7 object(s) allocated from:
//    #0 0x7f29f0f7807b in __interceptor_malloc libsanitizer/asan/asan_malloc_linux.cpp:145
//    #1 0x7f2945584a64  (<unknown module>)
//
// etc.
#if LEGATE_DEFINED(LEGATE_HAS_ASAN)
#define LegionPreInit DISABLED_LegionPreInit  // NOLINT(readability-identifier-naming)
#endif

TEST_F(RuntimeInitUnit, LegionPreInit)
{
  int argc                 = 1;
  const char* dummy_argv[] = {"legate-placeholder-binary-name", nullptr};
  // Realm won't modify the existing strings, but nevertheless they require a char*
  char** argv = const_cast<char**>(dummy_argv);

  // If Realm finds anything in REALM_DEFAULT_ARGS, it will copy it onto the command line, right
  // after the (fake) program name. So at exit we should free everything except the first token.
  LEGATE_SCOPE_GUARD(
    if (argv != dummy_argv) {
      for (int i = 1; i < argc; ++i) {
        std::free(argv[i]);
      }
      std::free(static_cast<void*>(argv));
    });

  ASSERT_NO_THROW(Legion::Runtime::initialize(&argc, &argv, /*filter=*/false, /*parse=*/false));
  ASSERT_EQ(Legion::Runtime::start(argc, argv, /* background */ true), 0);
  // Cannot start the runtime if Legion has already started
  ASSERT_THROW(legate::start(), std::runtime_error);
  ASSERT_FALSE(legate::has_started());
  ASSERT_EQ(Legion::Runtime::wait_for_shutdown(), 0);
}

TEST_F(RuntimeInitUnit, MultiInit)
{
  // OK to call start() multiple times
  ASSERT_NO_THROW(legate::start());
  ASSERT_TRUE(legate::has_started());
  ASSERT_NO_THROW(legate::start());
  ASSERT_TRUE(legate::has_started());
  ASSERT_EQ(legate::finish(), 0);
  ASSERT_TRUE(legate::has_finished());
  ASSERT_FALSE(legate::has_started());
}

TEST_F(RuntimeInitUnit, Restart)
{
  ASSERT_NO_THROW(legate::start());
  ASSERT_TRUE(legate::has_started());
  ASSERT_EQ(legate::finish(), 0);
  ASSERT_TRUE(legate::has_finished());
  ASSERT_FALSE(legate::has_started());
  // Cannot restart the runtime
  ASSERT_THROW(legate::start(), std::runtime_error);
  ASSERT_TRUE(legate::has_finished());
  ASSERT_FALSE(legate::has_started());
}

}  // namespace runtime_init_test
