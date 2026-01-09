/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/detail/env.h>
#include <legate/utilities/detail/zstring_view.h>

#include <fmt/format.h>

#include <gtest/gtest.h>

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <random>
#include <string>
#include <utilities/env.h>
#include <utilities/utilities.h>

namespace environment_variable_test {

namespace {

namespace detail {

template <typename T>
[[nodiscard]] T get_random_integral_value()
{
  static_assert(std::is_integral_v<T>);
  std::random_device dev{};
  std::mt19937 rng{dev()};
  std::uniform_int_distribution<T> dist{};

  return dist(rng);
}

template <typename T>
T get_random_value() = delete;

template <>
[[nodiscard]] std::int64_t get_random_value()
{
  return get_random_integral_value<std::int64_t>();
}

template <>
[[nodiscard]] std::uint32_t get_random_value()
{
  return get_random_integral_value<std::uint32_t>();
}

template <>
[[nodiscard]] bool get_random_value()
{
  return get_random_value<std::int64_t>() % 2 == 0;
}

template <>
[[nodiscard]] std::string get_random_value()
{
  static constexpr std::string_view alphanum =
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz";
  constexpr auto len = 32;

  std::string s;

  s.reserve(len);
  for (int i = 0; i < len; ++i) {
    s.push_back(alphanum.at(get_random_value<std::uint32_t>() % alphanum.size()));
  }

  return s;
}

}  // namespace detail

using EnvironTypeList = ::testing::Types<bool, std::int64_t, std::uint32_t, std::string>;

class NameGenerator {
 public:
  template <typename T>
  static std::string GetName(int)  // NOLINT(readability-identifier-naming)
  {
    if constexpr (std::is_same_v<T, bool>) {
      return "bool";
    } else if constexpr (std::is_same_v<T, std::int64_t>) {
      return "std::int64_t";
    } else if constexpr (std::is_same_v<T, std::uint32_t>) {
      return "std::uint32_t";
    } else if constexpr (std::is_same_v<T, std::string>) {
      return "std::string";
    }
    return "unknown";
  }
};

template <typename T>
class Environ : public DefaultFixture {
 public:
  [[nodiscard]] static std::pair<T, std::string> get_random_value()
  {
    auto val     = detail::get_random_value<T>();
    auto str_val = [&] {
      if constexpr (std::is_same_v<T, std::string>) {
        return val;
      } else {
        return std::to_string(val);
      }
    }();

    return {std::move(val), std::move(str_val)};
  }

  [[nodiscard]] legate::detail::EnvironmentVariable<T> make_env_var() const
  {
    const auto name = make_env_var_name_();
    auto env_var    = legate::detail::EnvironmentVariable<T>{name};

    EXPECT_EQ(name, static_cast<legate::detail::ZStringView>(env_var));

    const auto cur_val = legate::test::Environment::get_env_var(env_var.data());

    EXPECT_EQ(cur_val, nullptr) << "Someone else has defined " << name
                                << " in the environment: " << cur_val;
    return env_var;
  }

 private:
  [[nodiscard]] legate::detail::ZStringView make_env_var_name_() const
  {
    auto name = detail::get_random_value<std::string>();

    name += "_";
    name += NameGenerator::GetName<T>(0);
    std::transform(
      name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::toupper(c); });
    // So that is stays alive for the test, since EnvironmentVariable takes a string_view
    return env_vars_.emplace_back(std::move(name));
  }

  mutable std::deque<std::string> env_vars_{};
};

}  // namespace

TYPED_TEST_SUITE(Environ, EnvironTypeList, NameGenerator);

// ==========================================================================================

TYPED_TEST(Environ, Set)
{
  const auto ENV_VAR        = this->make_env_var();
  const auto [val, str_val] = this->get_random_value();

  ASSERT_EQ(legate::test::Environment::get_env_var(ENV_VAR.data()), nullptr);

  ENV_VAR.set(val);

  const auto new_val = legate::test::Environment::get_env_var(ENV_VAR.data());

  ASSERT_NE(new_val, nullptr);
  ASSERT_STREQ(new_val, str_val.c_str());

  legate::test::Environment::unset_env_var(ENV_VAR.data());
}

TYPED_TEST(Environ, SetGet)
{
  const auto ENV_VAR        = this->make_env_var();
  const auto [val, str_val] = this->get_random_value();

  ASSERT_EQ(legate::test::Environment::get_env_var(ENV_VAR.data()), nullptr);

  ENV_VAR.set(val);

  const auto val_after = ENV_VAR.get();

  ASSERT_TRUE(val_after.has_value());
  ASSERT_EQ(*val_after, val);  // NOLINT(bugprone-unchecked-optional-access)

  legate::test::Environment::unset_env_var(ENV_VAR.data());
}

TYPED_TEST(Environ, GetWhenSet)
{
  const auto ENV_VAR        = this->make_env_var();
  const auto [val, str_val] = this->get_random_value();

  legate::test::Environment::set_env_var(ENV_VAR.data(), str_val.c_str(), true);

  const auto ret = ENV_VAR.get();

  ASSERT_TRUE(ret.has_value());
  ASSERT_EQ(*ret, val);  // NOLINT(bugprone-unchecked-optional-access)

  legate::test::Environment::unset_env_var(ENV_VAR.data());
}

TYPED_TEST(Environ, GetUnset)
{
  const auto ENV_VAR = this->make_env_var();
  const auto ret     = ENV_VAR.get();

  ASSERT_FALSE(ret.has_value());
}

TYPED_TEST(Environ, GetDefaultWhenSet)
{
  const auto ENV_VAR            = this->make_env_var();
  const auto [set_val, str_val] = this->get_random_value();
  auto [default_val, _]         = this->get_random_value();
  constexpr auto max_its        = 100;

  // This loop mainly exists for the bool case, where it is extremely easy to get 2 values that
  // are the same.
  for (int i = 0; i < max_its; ++i) {
    if (default_val != set_val) {
      break;
    }
    std::tie(default_val, std::ignore) = this->get_random_value();
  }
  ASSERT_NE(set_val, default_val) << "Value must not equal default value: " << default_val;

  legate::test::Environment::set_env_var(ENV_VAR.data(), str_val.c_str(), true);

  const auto ret = ENV_VAR.get(default_val);

  // Should equal the set value, not the default
  ASSERT_EQ(ret, set_val);

  legate::test::Environment::unset_env_var(ENV_VAR.data());
}

TYPED_TEST(Environ, GetDefaultUnset)
{
  const auto ENV_VAR  = this->make_env_var();
  const auto [val, _] = this->get_random_value();
  const auto ret      = ENV_VAR.get(val);

  ASSERT_EQ(ret, val);

  legate::test::Environment::unset_env_var(ENV_VAR.data());
}

TYPED_TEST(Environ, GetTestWhenSet)
{
  const auto ENV_VAR            = this->make_env_var();
  const auto [set_val, str_val] = this->get_random_value();
  auto [default_val, _]         = this->get_random_value();
  constexpr auto max_its        = 100;

  // This loop mainly exists for the bool case, where it is extremely easy to get 2 values that
  // are the same.
  for (int i = 0; i < max_its; ++i) {
    if (default_val != set_val) {
      break;
    }
    std::tie(default_val, std::ignore) = this->get_random_value();
  }
  ASSERT_NE(set_val, default_val) << "Value must not equal default value: " << default_val;

  const auto tmp = legate::test::Environment::temporary_env_var("LEGATE_TEST", "1", true);

  legate::test::Environment::set_env_var(ENV_VAR.data(), str_val.c_str(), true);

  // If default_val and test_val are the same, and default_val != set_val, then we can be sure
  // we are getting the right value below
  const auto test_val = default_val;
  const auto ret      = ENV_VAR.get(default_val, test_val);

  // Should equal the set value, not the default or test value
  ASSERT_EQ(ret, set_val);

  legate::test::Environment::unset_env_var(ENV_VAR.data());
}

TYPED_TEST(Environ, GetTestUnset)
{
  const auto ENV_VAR          = this->make_env_var();
  const auto [default_val, _] = this->get_random_value();
  auto [test_val, _2]         = this->get_random_value();
  constexpr auto max_its      = 100;

  // This loop mainly exists for the bool case, where it is extremely easy to get 2 values that
  // are the same.
  for (int i = 0; i < max_its; ++i) {
    if (default_val != test_val) {
      break;
    }
    std::tie(test_val, std::ignore) = this->get_random_value();
  }
  ASSERT_NE(default_val, test_val) << "Value must not equal test value: " << test_val;

  const auto tmp = legate::test::Environment::temporary_env_var("LEGATE_TEST", "1", true);
  const auto ret = ENV_VAR.get(default_val, test_val);

  // Should equal the TEST value, not the default
  ASSERT_EQ(ret, test_val);

  legate::test::Environment::unset_env_var(ENV_VAR.data());
}

namespace {

class Environ2 : public DefaultFixture {};

}  // namespace

TEST_F(Environ2, BadValueInvalidArgument)
{
  constexpr auto var = legate::detail::EnvironmentVariable<std::uint32_t>{"FOO_BAR_BAZ"};
  const auto tmp     = legate::test::Environment::temporary_env_var("FOO_BAR_BAZ", "true", true);

  ASSERT_THROW(static_cast<void>(var.get()), std::invalid_argument);
}

TEST_F(Environ2, BadValueOutOfRange)
{
  constexpr auto var = legate::detail::EnvironmentVariable<std::uint32_t>{"FOO_BAR_BAZ"};
  // The value is out of range for uin32_t
  static_assert(std::numeric_limits<std::uint32_t>::max() <
                std::numeric_limits<std::uint64_t>::max());
  const auto val = std::to_string(std::numeric_limits<std::uint64_t>::max());
  const auto tmp = legate::test::Environment::temporary_env_var("FOO_BAR_BAZ", val.c_str(), true);

  ASSERT_THROW(static_cast<void>(var.get()), std::out_of_range);
}

}  // namespace environment_variable_test
