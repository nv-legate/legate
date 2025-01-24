/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <legate/utilities/detail/env.h>

#include <gtest/gtest.h>

#include <cstdlib>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

namespace legate::test {

class Environment {
 public:
  Environment() = delete;

  [[nodiscard]] static std::unique_lock<std::mutex> environment_lock()
  {
    static std::mutex mut{};

    return std::unique_lock<std::mutex>{mut};
  }

  static void set_env_var(const char* name, const char* value, bool overwrite)
  {
    const auto _ = environment_lock();

    ASSERT_EQ(::setenv(name, value, overwrite ? 1 : 0), 0)
      << "failed to set " << name << " to " << value << " overwrite: " << overwrite;
  }

  [[nodiscard]] static const char* get_env_var(const char* name)
  {
    const auto _ = environment_lock();

    return std::getenv(name);
  }

  static void unset_env_var(const char* name)
  {
    const auto _ = environment_lock();

    ASSERT_EQ(::unsetenv(name), 0) << "failed to unset " << name;
  }

  class TemporaryEnvVar {
   public:
    TemporaryEnvVar(std::string name, std::nullptr_t) : TemporaryEnvVar{std::move(name)}
    {
      Environment::unset_env_var(name_.c_str());
    }

    TemporaryEnvVar(std::string name, const char* value, bool overwrite)
      : TemporaryEnvVar{std::move(name)}
    {
      if (!value) {
        throw std::invalid_argument{"value must not be NULL"};
      }
      Environment::set_env_var(name_.c_str(), value, overwrite);
    }

    ~TemporaryEnvVar()
    {
      if (prev_value_.has_value()) {
        Environment::set_env_var(name_.c_str(), prev_value_->c_str(), true);
      } else {
        Environment::unset_env_var(name_.c_str());
      }
    }

   private:
    explicit TemporaryEnvVar(std::string name)
      : name_{std::move(name)}, prev_value_{[&]() -> std::optional<std::string> {
          if (const auto* cur_val = Environment::get_env_var(name_.c_str())) {
            return cur_val;
          }
          return std::nullopt;
        }()}
    {
    }

    std::string name_{};
    std::optional<std::string> prev_value_{};
  };

  [[nodiscard]] static TemporaryEnvVar temporary_env_var(std::string name,
                                                         const char* value,
                                                         bool overwrite)
  {
    return {std::move(name), value, overwrite};
  }

  template <typename T>
  [[nodiscard]] static TemporaryEnvVar temporary_env_var(
    const legate::detail::EnvironmentVariable<T>& name, const char* value, bool overwrite)
  {
    return temporary_env_var(
      static_cast<legate::detail::ZStringView>(name).to_string(), value, overwrite);
  }

  [[nodiscard]] static TemporaryEnvVar temporary_cleared_env_var(std::string name)
  {
    return {std::move(name), nullptr};
  }

  template <typename T>
  [[nodiscard]] static TemporaryEnvVar temporary_cleared_env_var(
    const legate::detail::EnvironmentVariable<T>& name)
  {
    return temporary_cleared_env_var(static_cast<legate::detail::ZStringView>(name).to_string());
  }
};

}  // namespace legate::test
