/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/detail/traced_exception.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <non_reentrant/wo_runtime/exception/common.h>
#include <stdexcept>
#include <string>
#include <string_view>

namespace traced_exception_test {

namespace {

class UserException : public std::exception {
 public:
  explicit UserException(std::string w) : what_{std::move(w)} {}

  [[nodiscard]] const char* what() const noexcept override { return what_.c_str(); }

 private:
  std::string what_{};
};

template <typename E>
void throw_exn()
{
  throw legate::detail::TracedException<E>{"oh no"};
}

}  // namespace

using ExceptionConstructTypeList =
  ::testing::Types<std::runtime_error, std::logic_error, UserException>;

template <typename T>
class TracedExceptionConstruct : public TracedExceptionFixture {};

TYPED_TEST_SUITE(TracedExceptionConstruct, ExceptionConstructTypeList, );

TYPED_TEST(TracedExceptionConstruct, Basic) { ASSERT_THROW(throw_exn<TypeParam>(), TypeParam); }

class TracedExceptionUnit : public TracedExceptionFixture {};

TEST_F(TracedExceptionUnit, Stacktrace)
{
  constexpr auto orig_msg = std::string_view{"a very important message"};
  using exn_type          = std::logic_error;
  const auto exn          = legate::detail::TracedException<std::logic_error>{orig_msg.data()};

  ASSERT_THAT(
    exn.what(),
    MatchesStackTrace(
      std::array{std::cref(typeid(exn_type))}, std::array{orig_msg}, std::array{__FILE__}));
}

TEST_F(TracedExceptionUnit, Nested)
{
  constexpr auto child_msg  = std::string_view{"child message"};
  using child_type          = std::logic_error;
  constexpr auto parent_msg = std::string_view{"parent message"};
  using parent_type         = std::overflow_error;

  try {
    try {
      throw legate::detail::TracedException<child_type>{child_msg.data()};
    } catch (const std::exception&) {
      throw legate::detail::TracedException<parent_type>{parent_msg.data()};
    }
  } catch (const std::exception& e) {
    ASSERT_THAT(
      e.what(),
      MatchesStackTrace(std::array{std::cref(typeid(child_type)), std::cref(typeid(parent_type))},
                        std::array{child_msg, parent_msg},
                        std::array{__FILE__, __FILE__}));
  } catch (...) {
    GTEST_FAIL() << "Failed to catch traced exception";
  }
}

using TracedExceptionDeathTest = TracedExceptionUnit;

class TracedNonStdException : public legate::detail::TracedExceptionBase {
 public:
  TracedNonStdException() : TracedExceptionBase{std::make_exception_ptr(NonStdException{}), 0} {}
};

class TracedNestedNonStdException : public legate::detail::TracedExceptionBase {
 public:
  TracedNestedNonStdException()
    : TracedExceptionBase{std::make_exception_ptr(TracedNonStdException{}), 0}
  {
  }
};

TEST_F(TracedExceptionDeathTest, TracedWhat)
{
  const auto exn = TracedNonStdException{};
  ASSERT_EXIT(
    { static_cast<void>(exn.traced_what()); },
    ::testing::KilledBySignal{SIGABRT},
    "Original exception not derived from std::exception");
}

TEST_F(TracedExceptionDeathTest, NestedTracedWhat)
{
  constexpr auto throw_nested_nonstd_expt = []() {
    try {
      try {
        throw NonStdException{};
      } catch (...) {
        throw legate::detail::TracedException<std::logic_error>{"outer error"};
      }
    } catch (const legate::detail::TracedExceptionBase& e) {
      static_cast<void>(e.traced_what());
    }
  };

  ASSERT_EXIT(throw_nested_nonstd_expt(),
              ::testing::KilledBySignal{SIGABRT},
              "Nested exception not derived from std::exception");
}

TEST_F(TracedExceptionDeathTest, RawWhatSV)
{
  const auto exn = TracedNonStdException{};
  ASSERT_EXIT(
    { static_cast<void>(exn.raw_what_sv()); },
    ::testing::KilledBySignal{SIGABRT},
    "Original exception not derived from std::exception");
}

TEST_F(TracedExceptionDeathTest, NestedRawWhatSV)
{
  const auto exn = TracedNestedNonStdException{};
  ASSERT_EXIT(
    { static_cast<void>(exn.raw_what_sv()); },
    ::testing::KilledBySignal{SIGABRT},
    "Exception must not be a traced exception");
}

}  // namespace traced_exception_test
