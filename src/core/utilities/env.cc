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

#include "core/utilities/env.h"

#include "core/utilities/assert.h"  // LEGATE_LIKELY()

#include <cerrno>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace legate::detail {

namespace {

template <typename T>
using ImplFunctionType = std::optional<T> (*)(std::string_view);

[[nodiscard]] std::lock_guard<std::mutex> ENVIRONMENT_LOCK()
{
  static std::mutex mut{};

  return std::lock_guard<std::mutex>{mut};
}

template <typename T>
[[nodiscard]] std::optional<T> read_env(std::string_view) = delete;

template <>
[[nodiscard]] std::optional<std::int64_t> read_env(std::string_view variable)
{
  if (variable.empty()) {
    throw std::invalid_argument{"Environment variable name is empty"};
  }

  auto parsed_val = [&]() -> std::optional<std::int64_t> {
    const auto _      = ENVIRONMENT_LOCK();
    const char* value = [&] {
      if (LEGATE_LIKELY(variable.back())) {
        return std::getenv(variable.data());
      }
      // std::string will null-terminate a non-null-terminated string_view for us
      return std::getenv(std::string{variable}.c_str());
    }();

    if (!value) {
      return std::nullopt;
    }

    std::int64_t ret    = 0;
    const auto value_sv = std::string_view{value};
    if (auto [ptr, ec] = std::from_chars(value_sv.begin(), value_sv.end(), ret);
        ec != std::errc{}) {
      throw std::runtime_error{std::make_error_code(ec).message()};
    }
    // We need to hold the environment lock until here since we are still reading from it in
    // from_chars().
    return ret;
  }();

  if (parsed_val.has_value() && (parsed_val < 0)) {
    std::stringstream ss;

    ss << "Invalid value for config value \"" << variable << "\": " << *parsed_val
       << ". Value must not be negative.";
    throw std::invalid_argument{std::move(ss).str()};
  }
  return parsed_val;
}

template <>
[[nodiscard]] std::optional<bool> read_env(std::string_view variable)
{
  if (const auto v = read_env<std::int64_t>(std::move(variable)); v.has_value()) {
    return *v > 0;
  }
  return std::nullopt;
}

template <>
[[nodiscard]] std::optional<std::uint32_t> read_env(std::string_view variable)
{
  if (const auto v = read_env<std::int64_t>(std::move(variable)); v.has_value()) {
    return static_cast<std::uint32_t>(*v);
  }
  return std::nullopt;
}

template <typename T>
[[nodiscard]] auto read_env_with_defaults(ImplFunctionType<T> read_env_impl_fn,
                                          std::string_view variable,
                                          T default_value,
                                          std::optional<T> test_value) -> T
{
  if (auto&& value = read_env_impl_fn(std::move(variable)); value.has_value()) {
    return *std::move(value);
  }
  // Can save another env access if we are only given a default value
  if (!test_value.has_value()) {
    return default_value;
  }
  // Don't use the "default values" getter here, since that would lead to infinite recursive
  // loop.
  return LEGATE_TEST.get().value_or(false) ? *test_value : default_value;
}

}  // namespace

// ==========================================================================================

void EnvironmentVariableBase::set(std::string_view value, bool overwrite) const
{
  const auto ret = [&]() {
    const auto _ = ENVIRONMENT_LOCK();

    // Reset this here so that we make sure any modification originates from setenv()
    errno = 0;
    return setenv(data(), value.data(), overwrite ? 1 : 0);
  }();
  if (LEGATE_UNLIKELY(ret)) {
    // In case stringstream writes to errno before we have the chance to strerror() it.
    const auto errno_save = errno;
    std::stringstream ss;

    ss << "setenv(" << static_cast<std::string_view>(*this) << ", " << value
       << ") failed with exit code: " << ret << ": " << std::strerror(errno_save);
    throw std::runtime_error{std::move(ss).str()};
  }
}

// ==========================================================================================

std::optional<bool> EnvironmentVariable<bool>::get() const { return read_env<bool>(*this); }

bool EnvironmentVariable<bool>::get(bool default_value, std::optional<bool> test_value) const
{
  return read_env_with_defaults(read_env<>, *this, default_value, std::move(test_value));
}

void EnvironmentVariable<bool>::set(bool value, bool overwrite) const
{
  EnvironmentVariableBase::set(value ? "1" : "0", overwrite);
}

// ==========================================================================================

std::optional<std::int64_t> EnvironmentVariable<std::int64_t>::get() const
{
  return read_env<std::int64_t>(*this);
}

std::int64_t EnvironmentVariable<std::int64_t>::get(std::int64_t default_value,
                                                    std::optional<std::int64_t> test_value) const
{
  return read_env_with_defaults(read_env<>, *this, default_value, std::move(test_value));
}

void EnvironmentVariable<std::int64_t>::set(std::int64_t value, bool overwrite) const
{
  EnvironmentVariableBase::set(std::to_string(value), overwrite);
}

// ==========================================================================================

std::optional<std::uint32_t> EnvironmentVariable<std::uint32_t>::get() const
{
  return read_env<std::uint32_t>(*this);
}

std::uint32_t EnvironmentVariable<std::uint32_t>::get(std::uint32_t default_value,
                                                      std::optional<std::uint32_t> test_value) const
{
  return read_env_with_defaults(read_env<>, *this, default_value, std::move(test_value));
}

void EnvironmentVariable<std::uint32_t>::set(std::uint32_t value, bool overwrite) const
{
  EnvironmentVariableBase::set(std::to_string(value), overwrite);
}

}  // namespace legate::detail
