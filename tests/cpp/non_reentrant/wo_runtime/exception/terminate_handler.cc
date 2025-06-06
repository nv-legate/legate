/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/detail/traced_exception.h>

#include <fmt/format.h>
#include <fmt/std.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <exception>
#include <non_reentrant/wo_runtime/exception/common.h>
#include <stdexcept>
#include <utilities/utilities.h>

namespace traced_exception_test {

class TerminateHandlerUnit : public DefaultFixture {};

TEST_F(TerminateHandlerUnit, Install)
{
  ASSERT_TRUE(legate::detail::install_terminate_handler());
  // Only first invocation has any effect
  ASSERT_FALSE(legate::detail::install_terminate_handler());
  ASSERT_FALSE(legate::detail::install_terminate_handler());
  ASSERT_FALSE(legate::detail::install_terminate_handler());
  ASSERT_FALSE(legate::detail::install_terminate_handler());
}

class TerminateHandlerDeathTest : public TracedExceptionFixture {
  void SetUp() override
  {
    static_cast<void>(legate::detail::install_terminate_handler());
    TracedExceptionFixture::SetUp();
  }
};

namespace {

template <typename T>
void throw_and_terminate(const char* text)
{
  try {
    throw T{text};  // legate-lint: no-trace
  } catch (...) {
    std::terminate();
  }
  FAIL() << "Must not reach this point, failed to terminate";
}

}  // namespace

TEST_F(TerminateHandlerDeathTest, Basic)
{
  constexpr auto text = "An exception";
  using exn_ty        = std::runtime_error;

  ASSERT_EXIT(throw_and_terminate<exn_ty>(text),
              ::testing::KilledBySignal(SIGABRT),
              MatchesStackTrace(
                std::array{std::cref(typeid(exn_ty))}, std::array{text}, std::array{__FILE__}));
}

TEST_F(TerminateHandlerDeathTest, TracedException)
{
  constexpr auto text = "An exception";
  using inner_exn_ty  = std::runtime_error;
  using exn_ty        = legate::detail::TracedException<inner_exn_ty>;

  ASSERT_EXIT(
    throw_and_terminate<exn_ty>(text),
    ::testing::KilledBySignal(SIGABRT),
    MatchesStackTrace(
      std::array{std::cref(typeid(inner_exn_ty))}, std::array{text}, std::array{__FILE__}));
}

TEST_F(TerminateHandlerDeathTest, NonStdException)
{
  constexpr auto text = "An exception";
  using exn_ty        = NonStdException;

  ASSERT_EXIT(throw_and_terminate<exn_ty>(text), ::testing::KilledBySignal(SIGABRT), "");
}

TEST_F(TerminateHandlerDeathTest, Nested)
{
  constexpr auto throw_nested_exception = [](auto&& f) -> decltype(auto) {
    try {
      try {
        throw std::runtime_error{"BOTTOM"};  // legate-lint: no-trace
      } catch (const std::exception& e) {
        std::throw_with_nested(std::invalid_argument{"TOP"});
      }
    } catch (const std::exception& exn) {
      return f(exn);
    }
    [] { GTEST_FAIL() << "Must not reach this point"; }();
  };

  const auto& nested_type_id = throw_nested_exception(
    [](const std::exception& exn) -> const std::type_info& { return typeid(exn); });

  EXPECT_EXIT(throw_nested_exception([](const auto&) { std::terminate(); }),
              ::testing::KilledBySignal(SIGABRT),
              ::testing::AllOf(
                ::testing::HasSubstr("LEGATE ERROR: #0 std::runtime_error: BOTTOM"),
                ::testing::HasSubstr(fmt::format("LEGATE ERROR: #1 {}: TOP", nested_type_id))));
}

TEST_F(TerminateHandlerDeathTest, NestedNonStdException)
{
  constexpr auto throw_nested_exception = [](auto&& f) -> decltype(auto) {
    try {
      try {
        throw NonStdException{"BOTTOM"};
      } catch (...) {
        std::throw_with_nested(std::invalid_argument{"TOP"});
      }
    } catch (const std::exception& exn) {
      return f(exn);
    }
    [] { GTEST_FAIL() << "Must not reach this point"; }();
  };

  const auto& nested_type_id = throw_nested_exception(
    [](const std::exception& exn) -> const std::type_info& { return typeid(exn); });

  EXPECT_EXIT(throw_nested_exception([](const auto&) { std::terminate(); }),
              ::testing::KilledBySignal(SIGABRT),
              ::testing::AllOf(
                ::testing::HasSubstr(fmt::format("LEGATE ERROR: #0 {}: TOP", nested_type_id))));
}

}  // namespace traced_exception_test
