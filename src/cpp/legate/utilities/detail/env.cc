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

#include "legate/utilities/detail/env.h"

#include "legate/utilities/assert.h"             // LEGATE_LIKELY()
#include "legate/utilities/detail/formatters.h"  // to format ZStringView
#include "legate/utilities/detail/zstring_view.h"

#include <cerrno>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>

namespace legate::detail {

namespace {

[[nodiscard]] std::lock_guard<std::mutex> ENVIRONMENT_LOCK()
{
  static std::mutex mut{};

  return std::lock_guard<std::mutex>{mut};
}

template <typename T>
[[nodiscard]] std::optional<T> read_env_common(ZStringView variable)
{
  if (variable.empty()) {
    throw std::invalid_argument{"Environment variable name is empty"};
  }

  const auto _ = ENVIRONMENT_LOCK();
  const char* value =
    std::getenv(variable.data());  // NOLINT(bugprone-suspicious-stringview-data-usage)

  if (!value) {
    return std::nullopt;
  }

  if constexpr (std::is_integral_v<T> || std::is_floating_point_v<T>) {
    const auto value_sv = std::string_view{value};
    T ret{};

    if (const auto [_2, ec] = std::from_chars(value_sv.begin(), value_sv.end(), ret);
        ec != std::errc{}) {
      throw std::runtime_error{std::make_error_code(ec).message()};
    }
    // We need to hold the environment lock until here since we are still reading from it in
    // from_chars().
    return ret;
  } else {
    return value;
  }
}

template <typename T>
[[nodiscard]] std::optional<T> read_env(ZStringView) = delete;

template <>
[[nodiscard]] std::optional<std::string> read_env(ZStringView variable)
{
  return read_env_common<std::string>(variable);
}

template <>
[[nodiscard]] std::optional<std::int64_t> read_env(ZStringView variable)
{
  auto parsed_val = read_env_common<std::int64_t>(variable);

  if (parsed_val.has_value() && (*parsed_val < 0)) {
    throw std::invalid_argument{
      fmt::format("Invalid value for config value \"{}\": {}. Value must not be negative.",
                  variable,
                  *parsed_val)};
  }
  return parsed_val;
}

template <>
[[nodiscard]] std::optional<bool> read_env(ZStringView variable)
{
  if (const auto v = read_env<std::int64_t>(std::move(variable)); v.has_value()) {
    return *v > 0;
  }
  return std::nullopt;
}

template <>
[[nodiscard]] std::optional<std::uint32_t> read_env(ZStringView variable)
{
  if (const auto v = read_env<std::int64_t>(std::move(variable)); v.has_value()) {
    return static_cast<std::uint32_t>(*v);
  }
  return std::nullopt;
}

template <typename T, typename U = T>
[[nodiscard]] T read_env_with_defaults(std::optional<T> (*read_env_impl_fn)(ZStringView),
                                       ZStringView variable,
                                       U default_value,
                                       std::optional<U> test_value)
{
  if (auto&& value = read_env_impl_fn(std::move(variable)); value.has_value()) {
    return *std::move(value);
  }
  // Can save another env access if we are only given a default value
  if (!test_value.has_value()) {
    if constexpr (std::is_same_v<T, U>) {
      return default_value;
    } else {
      return T{std::move(default_value)};
    }
  }
  // Don't use the "default values" getter here, since that would lead to infinite recursive
  // loop
  auto&& ret = LEGATE_TEST.get().value_or(false) ? *std::move(test_value) : default_value;

  if constexpr (std::is_same_v<T, U>) {
    return ret;
  } else {
    return T{std::move(ret)};
  }
}

}  // namespace

// ==========================================================================================

void EnvironmentVariableBase::set_(std::string_view value, bool overwrite) const
{
  const auto ret = [&]() {
    const auto _ = ENVIRONMENT_LOCK();

    // Reset this here so that we make sure any modification originates from setenv()
    errno = 0;
    return ::setenv(data(),
                    value.data(),  // NOLINT(bugprone-suspicious-stringview-data-usage)
                    overwrite ? 1 : 0);
  }();
  if (LEGATE_UNLIKELY(ret)) {
    throw std::runtime_error{fmt::format("setenv({}, {}) failed with exit code: {}: {}",
                                         static_cast<ZStringView>(*this),
                                         value,
                                         ret,
                                         std::strerror(errno))};
  }
}

// ==========================================================================================

std::optional<bool> EnvironmentVariable<bool>::get() const { return read_env<bool>(*this); }

bool EnvironmentVariable<bool>::get(bool default_value, std::optional<bool> test_value) const
{
  return read_env_with_defaults(read_env<bool>, *this, default_value, std::move(test_value));
}

void EnvironmentVariable<bool>::set(bool value, bool overwrite) const
{
  EnvironmentVariableBase::set_(value ? "1" : "0", overwrite);
}

// ==========================================================================================

std::optional<std::int64_t> EnvironmentVariable<std::int64_t>::get() const
{
  return read_env<std::int64_t>(*this);
}

std::int64_t EnvironmentVariable<std::int64_t>::get(std::int64_t default_value,
                                                    std::optional<std::int64_t> test_value) const
{
  return read_env_with_defaults(
    read_env<std::int64_t>, *this, default_value, std::move(test_value));
}

void EnvironmentVariable<std::int64_t>::set(std::int64_t value, bool overwrite) const
{
  EnvironmentVariableBase::set_(std::to_string(value), overwrite);
}

// ==========================================================================================

std::optional<std::uint32_t> EnvironmentVariable<std::uint32_t>::get() const
{
  return read_env<std::uint32_t>(*this);
}

std::uint32_t EnvironmentVariable<std::uint32_t>::get(std::uint32_t default_value,
                                                      std::optional<std::uint32_t> test_value) const
{
  return read_env_with_defaults(
    read_env<std::uint32_t>, *this, default_value, std::move(test_value));
}

void EnvironmentVariable<std::uint32_t>::set(std::uint32_t value, bool overwrite) const
{
  EnvironmentVariableBase::set_(std::to_string(value), overwrite);
}

// ==========================================================================================

std::optional<std::string> EnvironmentVariable<std::string>::get() const
{
  return read_env<std::string>(*this);
}

std::string EnvironmentVariable<std::string>::get(std::string_view default_value,
                                                  std::optional<std::string_view> test_value) const
{
  return read_env_with_defaults(read_env<std::string>, *this, default_value, std::move(test_value));
}

void EnvironmentVariable<std::string>::set(std::string_view value, bool overwrite) const
{
  EnvironmentVariableBase::set_(std::move(value), overwrite);
}

}  // namespace legate::detail
