/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/argument_parsing/argument.h>

#include <argparse/argparse.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <utilities/utilities.h>

namespace test_argument {

using ArgumentTypes = ::testing::Types<std::int32_t, std::string, std::filesystem::path, float>;

namespace {

[[nodiscard]] std::string as_string(std::string s) { return s; }

[[nodiscard]] std::string as_string(const std::filesystem::path& p) { return p; }

[[nodiscard]] std::string as_string(std::int32_t v) { return std::to_string(v); }

[[nodiscard]] std::string as_string(float v) { return std::to_string(v); }

template <typename T>
class Values {};

template <>
class Values<std::int32_t> {
 public:
  [[nodiscard]] static constexpr std::int32_t default_value()
  {
    return 1234;  // NOLINT(readability-magic-numbers)
  }

  [[nodiscard]] static constexpr std::int32_t alternate_value()
  {
    return 4321;  // NOLINT(readability-magic-numbers)
  }

  [[nodiscard]] static constexpr std::int32_t second_alternate_value()
  {
    return 6789;  // NOLINT(readability-magic-numbers)
  }
};

template <>
class Values<float> {
 public:
  [[nodiscard]] static constexpr float default_value()
  {
    return 1.0F;  // NOLINT(readability-magic-numbers)
  }

  [[nodiscard]] static constexpr float alternate_value()
  {
    return 2.0F;  // NOLINT(readability-magic-numbers)
  }

  [[nodiscard]] static constexpr float second_alternate_value()
  {
    return 3.0F;  // NOLINT(readability-magic-numbers)
  }
};

template <>
class Values<std::string> {
 public:
  [[nodiscard]] static std::string default_value() { return "foo,bar,baz"; };

  [[nodiscard]] static std::string alternate_value() { return "baz,bar,foo"; };

  [[nodiscard]] static std::string second_alternate_value() { return "qux,quux,bop"; };
};

template <>
class Values<std::filesystem::path> {
 public:
  [[nodiscard]] static std::filesystem::path default_value() { return "foo/bar/baz"; }

  [[nodiscard]] static std::filesystem::path alternate_value() { return "baz/bar/foo"; }

  [[nodiscard]] static std::filesystem::path second_alternate_value() { return "qux/quux/bop"; }
};

}  // namespace

template <typename T>
class ArgumentUnit : public DefaultFixture {
 public:
  void SetUp() override
  {
    DefaultFixture::SetUp();

    ASSERT_NE(defaults.default_value(), defaults.alternate_value());
    ASSERT_NE(defaults.second_alternate_value(), defaults.alternate_value());
  }

  std::shared_ptr<argparse::ArgumentParser> parser{
    std::make_shared<argparse::ArgumentParser>("program")};
  Values<T> defaults{};
};

TYPED_TEST_SUITE(ArgumentUnit, ArgumentTypes, );

TYPED_TEST(ArgumentUnit, DefaultConstruct)
{
  const auto flag          = std::string{"--foo"};
  const auto default_value = this->defaults.default_value();
  auto arg                 = legate::detail::Argument<TypeParam>{this->parser, flag, default_value};

  ASSERT_EQ(arg.flag(), flag);
  ASSERT_EQ(arg.value(), default_value);
  ASSERT_EQ(arg.value_mut(), default_value);
}

TYPED_TEST(ArgumentUnit, Mutate)
{
  const auto flag          = std::string{"--foo"};
  const auto default_value = this->defaults.default_value();
  auto arg                 = legate::detail::Argument<TypeParam>{this->parser, flag, default_value};

  ASSERT_EQ(arg.flag(), flag);
  ASSERT_EQ(arg.value(), default_value);
  ASSERT_EQ(arg.value_mut(), default_value);

  const auto alternate_value = this->defaults.alternate_value();

  arg.value_mut() = alternate_value;
  ASSERT_EQ(arg.value(), alternate_value);
  ASSERT_EQ(arg.value_mut(), alternate_value);
  ASSERT_NE(arg.value(), default_value);
  ASSERT_NE(arg.value_mut(), default_value);
}

TYPED_TEST(ArgumentUnit, SetFlag)
{
  const auto flag          = std::string{"--foo"};
  const auto default_value = this->defaults.default_value();
  const auto set_value     = this->defaults.alternate_value();
  auto arg                 = legate::detail::Argument<TypeParam>{this->parser, flag, default_value};
  auto& argparse_arg =
    this->parser->add_argument(flag).default_value(arg.value()).store_into(arg.value_mut());

  ASSERT_EQ(&argparse_arg, &arg.argparse_argument());
  ASSERT_FALSE(arg.was_set());
  this->parser->parse_args({"program", flag, as_string(set_value)});
  ASSERT_TRUE(arg.was_set());
  ASSERT_EQ(arg.value(), set_value);
  ASSERT_EQ(arg.value_mut(), set_value);
  ASSERT_NE(arg.value(), default_value);
  ASSERT_NE(arg.value_mut(), default_value);
}

TYPED_TEST(ArgumentUnit, DontSetFlag)
{
  const auto flag          = std::string{"--foo"};
  const auto default_value = this->defaults.default_value();
  const auto set_value     = this->defaults.alternate_value();
  auto arg                 = legate::detail::Argument<TypeParam>{this->parser, flag, default_value};
  auto& argparse_arg =
    this->parser->add_argument(flag).default_value(arg.value()).store_into(arg.value_mut());

  ASSERT_EQ(&argparse_arg, &arg.argparse_argument());
  ASSERT_FALSE(arg.was_set());
  this->parser->parse_args({"program" /* note, no flag */});
  ASSERT_FALSE(arg.was_set());
  ASSERT_NE(arg.value(), set_value);
  ASSERT_NE(arg.value_mut(), set_value);
  ASSERT_EQ(arg.value(), default_value);
  ASSERT_EQ(arg.value_mut(), default_value);
}

TYPED_TEST(ArgumentUnit, ActionSetFlag)
{
  const auto flag          = std::string{"--foo"};
  const auto default_value = this->defaults.default_value();
  const auto set_value     = this->defaults.alternate_value();
  const auto action_value  = this->defaults.second_alternate_value();
  auto arg                 = legate::detail::Argument<TypeParam>{this->parser, flag, default_value};
  auto action_called       = false;

  this->parser->add_argument(flag).default_value(arg.value()).store_into(arg.value_mut());
  arg.action([&](std::string_view value, legate::detail::Argument<TypeParam>* arg_value) {
    EXPECT_EQ(value, as_string(set_value));
    action_called          = true;
    arg_value->value_mut() = action_value;
    // We don't care if the value is moved (in fact, we don't want it to be)
    return action_value;  // NOLINT(performance-no-automatic-move)
  });
  ASSERT_FALSE(action_called);
  this->parser->parse_args({"program", flag, as_string(set_value)});
  ASSERT_TRUE(action_called);
  ASSERT_TRUE(arg.was_set());
  ASSERT_EQ(arg.value(), action_value);
  ASSERT_NE(arg.value(), default_value);
  ASSERT_NE(arg.value(), set_value);
}

TYPED_TEST(ArgumentUnit, ActionDontSetFlag)
{
  const auto flag          = std::string{"--foo"};
  const auto default_value = this->defaults.default_value();
  const auto set_value     = this->defaults.alternate_value();
  const auto action_value  = this->defaults.second_alternate_value();
  auto arg                 = legate::detail::Argument<TypeParam>{this->parser, flag, default_value};
  auto action_called       = false;

  this->parser->add_argument(flag).default_value(arg.value()).store_into(arg.value_mut());
  arg.action([&](std::string_view value, legate::detail::Argument<TypeParam>* arg_value) {
    EXPECT_EQ(value, as_string(set_value));
    action_called          = true;
    arg_value->value_mut() = action_value;
    // We don't care if the value is moved (in fact, we don't want it to be)
    return action_value;  // NOLINT(performance-no-automatic-move)
  });
  ASSERT_FALSE(action_called);
  this->parser->parse_args({"program" /* note, no flag */});
  ASSERT_FALSE(action_called);
  ASSERT_FALSE(arg.was_set());
  ASSERT_EQ(arg.value(), default_value);
  ASSERT_NE(arg.value(), action_value);
  ASSERT_NE(arg.value(), set_value);
}

TYPED_TEST(ArgumentUnit, ToString)
{
  const auto flag          = std::string{"--foo"};
  const auto default_value = this->defaults.default_value();
  auto arg                 = legate::detail::Argument<TypeParam>{this->parser, flag, default_value};

  std::stringstream ss;
  const auto ss_flags = ss.flags();
  std::stringstream ss_expected;

  ss << arg;
  ss_expected << "Argument(flag: --foo, value: " << default_value << ")";

  ASSERT_EQ(ss.str(), ss_expected.str());
  ASSERT_EQ(ss_flags, ss.flags());
}

using ArgumentUnitNegative = ArgumentUnit<std::string>;

class ThrowingBuffer : public std::streambuf {
 protected:
  int overflow(int) override { throw std::ios_base::failure{"Simulated failure"}; }
};

TEST_F(ArgumentUnitNegative, ToStringThrows)
{
  const auto flag  = std::string{"--foo"};
  const auto value = std::string{"abc"};
  auto arg         = legate::detail::Argument<std::string>{this->parser, flag, value};

  ThrowingBuffer buffer;
  std::ostream throwing_stream{&buffer};

  throwing_stream.exceptions(std::ios::badbit);

  const auto ss_flags = throwing_stream.flags();

  ASSERT_THAT(
    [&] { throwing_stream << arg; },
    ::testing::ThrowsMessage<std::ios_base::failure>(::testing::HasSubstr("Simulated failure")));
  ASSERT_EQ(ss_flags, throwing_stream.flags());
}

}  // namespace test_argument
