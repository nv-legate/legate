/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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

}  // namespace traced_exception_test
