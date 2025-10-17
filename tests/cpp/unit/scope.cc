/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/runtime/detail/runtime.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace scope_test {

using ScopeTest = DefaultFixture;

namespace {

constexpr std::int32_t MAGIC_PRIORITY1 = 42;
constexpr std::int32_t MAGIC_PRIORITY2 = 43;

constexpr legate::ExceptionMode MODE1 = legate::ExceptionMode::DEFERRED;
constexpr legate::ExceptionMode MODE2 = legate::ExceptionMode::IGNORED;

constexpr std::string_view MAGIC_PROVENANCE1 = "42";
constexpr std::string_view MAGIC_PROVENANCE2 = "43";

legate::mapping::Machine remove_last_proc(const legate::mapping::Machine& machine)
{
  return machine.slice(0, std::max(std::uint32_t{1}, machine.count() - 1));
}

}  // namespace

TEST_F(ScopeTest, BasicPriority)
{
  const auto old_priority = legate::Scope::priority();
  {
    const legate::Scope test_priority{MAGIC_PRIORITY1};

    EXPECT_EQ(legate::Scope::priority(), MAGIC_PRIORITY1);
  }
  EXPECT_EQ(legate::Scope::priority(), old_priority);
}

TEST_F(ScopeTest, BasicExceptionMode)
{
  const auto old_exception_mode = legate::Scope::exception_mode();
  {
    const legate::Scope test_exception_mode{MODE1};

    EXPECT_EQ(legate::Scope::exception_mode(), MODE1);
  }
  EXPECT_EQ(legate::Scope::exception_mode(), old_exception_mode);
}

TEST_F(ScopeTest, BasicProvenance)
{
  const auto old_provenance = legate::Scope::provenance();
  {
    const legate::Scope test_provenance{std::string{MAGIC_PROVENANCE1}};

    EXPECT_EQ(legate::Scope::provenance(), MAGIC_PROVENANCE1);
  }
  EXPECT_EQ(legate::Scope::provenance(), old_provenance);
}

TEST_F(ScopeTest, BasicMachine)
{
  const auto old_machine = legate::Scope::machine();
  {
    const auto sliced = remove_last_proc(legate::Scope::machine());
    const legate::Scope test_machine{sliced};

    EXPECT_EQ(legate::Scope::machine(), sliced);
  }
  EXPECT_EQ(legate::Scope::machine(), old_machine);
}

TEST_F(ScopeTest, NestedPriority)
{
  const auto old_priority = legate::Scope::priority();
  {
    const legate::Scope test_priority1{MAGIC_PRIORITY1};

    EXPECT_EQ(legate::Scope::priority(), MAGIC_PRIORITY1);
    {
      const legate::Scope test_priority2{MAGIC_PRIORITY2};

      EXPECT_EQ(legate::Scope::priority(), MAGIC_PRIORITY2);
    }
    EXPECT_EQ(legate::Scope::priority(), MAGIC_PRIORITY1);
  }
  EXPECT_EQ(legate::Scope::priority(), old_priority);
}

TEST_F(ScopeTest, NestedExceptionMode)
{
  const auto old_exception_mode = legate::Scope::exception_mode();
  {
    const legate::Scope test_exception_mode1{MODE1};

    EXPECT_EQ(legate::Scope::exception_mode(), MODE1);
    {
      const legate::Scope test_exception_mode2{MODE2};

      EXPECT_EQ(legate::Scope::exception_mode(), MODE2);
    }
    EXPECT_EQ(legate::Scope::exception_mode(), MODE1);
  }
  EXPECT_EQ(legate::Scope::exception_mode(), old_exception_mode);
}

TEST_F(ScopeTest, NestedProvenance)
{
  const auto old_provenance = legate::Scope::provenance();
  {
    const legate::Scope test_provenance1{std::string{MAGIC_PROVENANCE1}};

    EXPECT_EQ(legate::Scope::provenance(), MAGIC_PROVENANCE1);
    {
      const legate::Scope test_provenance2{std::string{MAGIC_PROVENANCE2}};

      EXPECT_EQ(legate::Scope::provenance(), MAGIC_PROVENANCE2);
    }
    EXPECT_EQ(legate::Scope::provenance(), MAGIC_PROVENANCE1);
  }
  EXPECT_EQ(legate::Scope::provenance(), old_provenance);
}

TEST_F(ScopeTest, NestedMachine)
{
  const auto old_machine = legate::Scope::machine();
  {
    const auto sliced1 = remove_last_proc(legate::Scope::machine());
    const legate::Scope test_machine1{sliced1};

    EXPECT_EQ(legate::Scope::machine(), sliced1);
    {
      const auto sliced2 = remove_last_proc(legate::Scope::machine());
      const legate::Scope test_machine2{sliced2};

      EXPECT_EQ(legate::Scope::machine(), sliced2);
    }
    EXPECT_EQ(legate::Scope::machine(), sliced1);
  }
  EXPECT_EQ(legate::Scope::machine(), old_machine);
}

TEST_F(ScopeTest, BasicChain)
{
  const auto old_priority       = legate::Scope::priority();
  const auto old_exception_mode = legate::Scope::exception_mode();
  const auto old_provenance     = legate::Scope::provenance();
  const auto old_machine        = legate::Scope::machine();
  {
    const auto sliced   = remove_last_proc(legate::Scope::machine());
    const auto test_all = legate::Scope{}
                            .with_priority(MAGIC_PRIORITY1)
                            .with_exception_mode(MODE1)
                            .with_provenance(std::string{MAGIC_PROVENANCE1})
                            .with_machine(sliced);

    EXPECT_EQ(legate::Scope::priority(), MAGIC_PRIORITY1);
    EXPECT_EQ(legate::Scope::exception_mode(), MODE1);
    EXPECT_EQ(legate::Scope::provenance(), MAGIC_PROVENANCE1);
    EXPECT_EQ(legate::Scope::machine(), sliced);
  }
  EXPECT_EQ(legate::Scope::priority(), old_priority);
  EXPECT_EQ(legate::Scope::exception_mode(), old_exception_mode);
  EXPECT_EQ(legate::Scope::provenance(), old_provenance);
  EXPECT_EQ(legate::Scope::machine(), old_machine);
}

TEST_F(ScopeTest, NestedChain)
{
  const auto old_priority       = legate::Scope::priority();
  const auto old_exception_mode = legate::Scope::exception_mode();
  const auto old_provenance     = legate::Scope::provenance();
  const auto old_machine        = legate::Scope::machine();
  {
    const auto sliced1   = remove_last_proc(legate::Scope::machine());
    const auto test_all1 = legate::Scope{}
                             .with_priority(MAGIC_PRIORITY1)
                             .with_exception_mode(MODE1)
                             .with_provenance(std::string{MAGIC_PROVENANCE1})
                             .with_machine(sliced1);

    EXPECT_EQ(legate::Scope::priority(), MAGIC_PRIORITY1);
    EXPECT_EQ(legate::Scope::exception_mode(), MODE1);
    EXPECT_EQ(legate::Scope::provenance(), MAGIC_PROVENANCE1);
    EXPECT_EQ(legate::Scope::machine(), sliced1);
    {
      const auto sliced2   = remove_last_proc(legate::Scope::machine());
      const auto test_all2 = legate::Scope{}
                               .with_priority(MAGIC_PRIORITY2)
                               .with_exception_mode(MODE2)
                               .with_provenance(std::string{MAGIC_PROVENANCE2})
                               .with_machine(sliced2);

      EXPECT_EQ(legate::Scope::priority(), MAGIC_PRIORITY2);
      EXPECT_EQ(legate::Scope::exception_mode(), MODE2);
      EXPECT_EQ(legate::Scope::provenance(), MAGIC_PROVENANCE2);
      EXPECT_EQ(legate::Scope::machine(), sliced2);
    }

    EXPECT_EQ(legate::Scope::priority(), MAGIC_PRIORITY1);
    EXPECT_EQ(legate::Scope::exception_mode(), MODE1);
    EXPECT_EQ(legate::Scope::provenance(), MAGIC_PROVENANCE1);
    EXPECT_EQ(legate::Scope::machine(), sliced1);
  }
  EXPECT_EQ(legate::Scope::priority(), old_priority);
  EXPECT_EQ(legate::Scope::exception_mode(), old_exception_mode);
  EXPECT_EQ(legate::Scope::provenance(), old_provenance);
  EXPECT_EQ(legate::Scope::machine(), old_machine);
}

TEST_F(ScopeTest, BasicSet)
{
  const auto old_priority       = legate::Scope::priority();
  const auto old_exception_mode = legate::Scope::exception_mode();
  const auto old_provenance     = legate::Scope::provenance();
  const auto old_machine        = legate::Scope::machine();
  {
    const auto sliced = remove_last_proc(legate::Scope::machine());
    auto test_all     = legate::Scope{};

    test_all.set_priority(MAGIC_PRIORITY1);
    test_all.set_exception_mode(MODE1);
    test_all.set_provenance(std::string{MAGIC_PROVENANCE1});
    test_all.set_machine(sliced);
    EXPECT_EQ(legate::Scope::priority(), MAGIC_PRIORITY1);
    EXPECT_EQ(legate::Scope::exception_mode(), MODE1);
    EXPECT_EQ(legate::Scope::provenance(), MAGIC_PROVENANCE1);
    EXPECT_EQ(legate::Scope::machine(), sliced);
  }
  EXPECT_EQ(legate::Scope::priority(), old_priority);
  EXPECT_EQ(legate::Scope::exception_mode(), old_exception_mode);
  EXPECT_EQ(legate::Scope::provenance(), old_provenance);
  EXPECT_EQ(legate::Scope::machine(), old_machine);
}

TEST_F(ScopeTest, NestedSet)
{
  const auto old_priority       = legate::Scope::priority();
  const auto old_exception_mode = legate::Scope::exception_mode();
  const auto old_provenance     = legate::Scope::provenance();
  const auto old_machine        = legate::Scope::machine();
  {
    const auto sliced1 = remove_last_proc(legate::Scope::machine());
    auto test_all1     = legate::Scope{};

    test_all1.set_priority(MAGIC_PRIORITY1);
    test_all1.set_exception_mode(MODE1);
    test_all1.set_provenance(std::string{MAGIC_PROVENANCE1});
    test_all1.set_machine(sliced1);
    EXPECT_EQ(legate::Scope::priority(), MAGIC_PRIORITY1);
    EXPECT_EQ(legate::Scope::exception_mode(), MODE1);
    EXPECT_EQ(legate::Scope::provenance(), MAGIC_PROVENANCE1);
    EXPECT_EQ(legate::Scope::machine(), sliced1);

    {
      const auto sliced2 = remove_last_proc(legate::Scope::machine());
      auto test_all2     = legate::Scope{};

      test_all2.set_priority(MAGIC_PRIORITY2);
      test_all2.set_exception_mode(MODE2);
      test_all2.set_provenance(std::string{MAGIC_PROVENANCE2});
      test_all2.set_machine(sliced2);
      EXPECT_EQ(legate::Scope::priority(), MAGIC_PRIORITY2);
      EXPECT_EQ(legate::Scope::exception_mode(), MODE2);
      EXPECT_EQ(legate::Scope::provenance(), MAGIC_PROVENANCE2);
      EXPECT_EQ(legate::Scope::machine(), sliced2);
    }

    EXPECT_EQ(legate::Scope::priority(), MAGIC_PRIORITY1);
    EXPECT_EQ(legate::Scope::exception_mode(), MODE1);
    EXPECT_EQ(legate::Scope::provenance(), MAGIC_PROVENANCE1);
    EXPECT_EQ(legate::Scope::machine(), sliced1);
  }
  EXPECT_EQ(legate::Scope::priority(), old_priority);
  EXPECT_EQ(legate::Scope::exception_mode(), old_exception_mode);
  EXPECT_EQ(legate::Scope::provenance(), old_provenance);
  EXPECT_EQ(legate::Scope::machine(), old_machine);
}

TEST_F(ScopeTest, DuplicatePriority1)
{
  legate::Scope test_priority{};

  test_priority.set_priority(MAGIC_PRIORITY1);
  EXPECT_THROW(test_priority.set_priority(MAGIC_PRIORITY2), std::invalid_argument);
}

TEST_F(ScopeTest, DuplicatePriority2)
{
  legate::Scope test_priority{MAGIC_PRIORITY1};

  EXPECT_THROW(test_priority.set_priority(MAGIC_PRIORITY2), std::invalid_argument);
}

TEST_F(ScopeTest, DuplicatePriority3)
{
  EXPECT_THROW(static_cast<void>(legate::Scope{MAGIC_PRIORITY1}.with_priority(MAGIC_PRIORITY2)),
               std::invalid_argument);
}

TEST_F(ScopeTest, DuplicateProvenance1)
{
  legate::Scope test_provenance{};

  test_provenance.set_provenance(std::string{MAGIC_PROVENANCE1});
  EXPECT_THROW(test_provenance.set_provenance(std::string{MAGIC_PROVENANCE2}),
               std::invalid_argument);
}

TEST_F(ScopeTest, DuplicateProvenance2)
{
  legate::Scope test_provenance{std::string{MAGIC_PROVENANCE1}};

  EXPECT_THROW(test_provenance.set_provenance(std::string{MAGIC_PROVENANCE2}),
               std::invalid_argument);
}

TEST_F(ScopeTest, DuplicateProvenance3)
{
  EXPECT_THROW(static_cast<void>(legate::Scope{std::string{MAGIC_PROVENANCE1}}.with_provenance(
                 std::string{MAGIC_PROVENANCE2})),
               std::invalid_argument);
}

TEST_F(ScopeTest, DuplicateMachine1)
{
  legate::Scope test_machine{};

  test_machine.set_machine(legate::Scope::machine());
  EXPECT_THROW(test_machine.set_machine(legate::Scope::machine()), std::invalid_argument);
}

TEST_F(ScopeTest, DuplicateMachine2)
{
  legate::Scope test_machine{legate::Scope::machine()};
  EXPECT_THROW(test_machine.set_machine(legate::Scope::machine()), std::invalid_argument);
}

TEST_F(ScopeTest, DuplicateMachine3)
{
  EXPECT_THROW(static_cast<void>(
                 legate::Scope{legate::Scope::machine()}.with_machine(legate::Scope::machine())),
               std::invalid_argument);
}

TEST_F(ScopeTest, StreamingSchedulingWindow)
{
  auto& runtime          = legate::detail::Runtime::get_runtime();
  const auto window_size = runtime.scope().scheduling_window_size();

  {
    const auto _ =
      legate::Scope{legate::ParallelPolicy{}.with_streaming(legate::StreamingMode::RELAXED)};
    const auto new_size = runtime.scope().scheduling_window_size();
    // Big window size is the arbitrary large window size chosen in
    // Scope::Impl::set_parallel_policy().
    constexpr auto BIG_WINDOW = 1024U;

    // Do GE because the tests may be run with a global window size that already exceeds our
    // big window.
    ASSERT_GE(new_size, BIG_WINDOW);
  }
  ASSERT_EQ(runtime.scope().scheduling_window_size(), window_size);
}

TEST_F(ScopeTest, StreamingSchedulingWindowNested)
{
  auto& runtime          = legate::detail::Runtime::get_runtime();
  const auto window_size = runtime.scope().scheduling_window_size();

  {
    const auto _ =
      legate::Scope{legate::ParallelPolicy{}.with_streaming(legate::StreamingMode::RELAXED)};
    const auto new_size = runtime.scope().scheduling_window_size();
    // Big window size is the arbitrary large window size chosen in
    // Scope::Impl::set_parallel_policy().
    constexpr auto BIG_WINDOW = 1024U;

    // Do GE because the tests may be run with a global window size that already exceeds our
    // big window.
    ASSERT_GE(new_size, BIG_WINDOW);
    {
      const auto _2 = legate::Scope{}.with_parallel_policy(
        legate::ParallelPolicy{}.with_streaming(legate::StreamingMode::RELAXED));
      const auto nested_new_size = runtime.scope().scheduling_window_size();

      ASSERT_GE(nested_new_size, BIG_WINDOW);
      ASSERT_EQ(nested_new_size, new_size);
    }
    ASSERT_EQ(runtime.scope().scheduling_window_size(), new_size);
  }
  ASSERT_EQ(runtime.scope().scheduling_window_size(), window_size);
}

}  // namespace scope_test
