/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/zstring_view.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace legate::detail {

class EnvironmentVariableBase {
 public:
  EnvironmentVariableBase() = delete;
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr EnvironmentVariableBase(ZStringView name) noexcept;

  // NOLINTNEXTLINE(google-explicit-constructor)
  [[nodiscard]] constexpr operator ZStringView() const noexcept;
  [[nodiscard]] constexpr const char* data() const noexcept;

 protected:
  void set_(std::string_view value, bool overwrite) const;

 private:
  ZStringView name_{};
};

// Need to define these here (instead of a .inl) so that they can be used in constexpr contexts
// below
constexpr EnvironmentVariableBase::EnvironmentVariableBase(ZStringView name) noexcept
  : name_{std::move(name)}
{
}

constexpr EnvironmentVariableBase::operator ZStringView() const noexcept { return name_; }

constexpr const char* EnvironmentVariableBase::data() const noexcept { return name_.data(); }

// ==========================================================================================

template <typename T>
class EnvironmentVariable {
 public:
  EnvironmentVariable() = delete;
};

template <>
class EnvironmentVariable<bool> : public EnvironmentVariableBase {
 public:
  using EnvironmentVariableBase::EnvironmentVariableBase;

  [[nodiscard]] std::optional<bool> get() const;
  [[nodiscard]] bool get(bool default_value, std::optional<bool> test_value = std::nullopt) const;
  void set(bool value, bool overwrite = true) const;
};

template <>
class EnvironmentVariable<std::int64_t> : public EnvironmentVariableBase {
 public:
  using EnvironmentVariableBase::EnvironmentVariableBase;

  [[nodiscard]] std::optional<std::int64_t> get() const;
  [[nodiscard]] std::int64_t get(std::int64_t default_value,
                                 std::optional<std::int64_t> test_value = std::nullopt) const;
  void set(std::int64_t value, bool overwrite = true) const;
};

template <>
class EnvironmentVariable<std::uint32_t> : public EnvironmentVariableBase {
 public:
  using EnvironmentVariableBase::EnvironmentVariableBase;

  [[nodiscard]] std::optional<std::uint32_t> get() const;
  [[nodiscard]] std::uint32_t get(std::uint32_t default_value,
                                  std::optional<std::uint32_t> test_value = std::nullopt) const;
  void set(std::uint32_t value, bool overwrite = true) const;
};

template <>
class EnvironmentVariable<std::string> : public EnvironmentVariableBase {
 public:
  using EnvironmentVariableBase::EnvironmentVariableBase;

  [[nodiscard]] std::optional<std::string> get() const;
  [[nodiscard]] std::string get(std::string_view default_value,
                                std::optional<std::string_view> test_value = std::nullopt) const;
  void set(std::string_view value, bool overwrite = true) const;
};

// ==========================================================================================

#define LEGATE_DEFINE_ENV_VAR(type, NAME) \
  inline constexpr EnvironmentVariable<type> NAME { #NAME }

LEGATE_DEFINE_ENV_VAR(bool, LEGATE_TEST);
LEGATE_DEFINE_ENV_VAR(bool, LEGATE_SHOW_USAGE);
LEGATE_DEFINE_ENV_VAR(bool, LEGATE_AUTO_CONFIG);
LEGATE_DEFINE_ENV_VAR(bool, LEGATE_SHOW_CONFIG);
LEGATE_DEFINE_ENV_VAR(bool, LEGATE_SHOW_PROGRESS);
LEGATE_DEFINE_ENV_VAR(bool, LEGATE_EMPTY_TASK);
LEGATE_DEFINE_ENV_VAR(bool, LEGATE_LOG_MAPPING);
LEGATE_DEFINE_ENV_VAR(bool, LEGATE_LOG_PARTITIONING);
LEGATE_DEFINE_ENV_VAR(bool, LEGATE_WARMUP_NCCL);
LEGATE_DEFINE_ENV_VAR(std::string, LEGION_DEFAULT_ARGS);
LEGATE_DEFINE_ENV_VAR(std::uint32_t, LEGATE_MAX_EXCEPTION_SIZE);
LEGATE_DEFINE_ENV_VAR(std::int64_t, LEGATE_MIN_CPU_CHUNK);
LEGATE_DEFINE_ENV_VAR(std::int64_t, LEGATE_MIN_GPU_CHUNK);
LEGATE_DEFINE_ENV_VAR(std::int64_t, LEGATE_MIN_OMP_CHUNK);
LEGATE_DEFINE_ENV_VAR(std::uint32_t, LEGATE_WINDOW_SIZE);
LEGATE_DEFINE_ENV_VAR(std::uint32_t, LEGATE_FIELD_REUSE_FRAC);
LEGATE_DEFINE_ENV_VAR(std::uint32_t, LEGATE_FIELD_REUSE_FREQ);
LEGATE_DEFINE_ENV_VAR(bool, LEGATE_CONSENSUS);
LEGATE_DEFINE_ENV_VAR(bool, LEGATE_DISABLE_MPI);
LEGATE_DEFINE_ENV_VAR(std::string, LEGATE_CONFIG);
LEGATE_DEFINE_ENV_VAR(std::string, LEGATE_MPI_WRAPPER);
LEGATE_DEFINE_ENV_VAR(std::string, LEGATE_CUDA_DRIVER);
LEGATE_DEFINE_ENV_VAR(bool, LEGATE_IO_USE_VFD_GDS);
LEGATE_DEFINE_ENV_VAR(std::string, REALM_UCP_BOOTSTRAP_MODE);

inline namespace experimental {

LEGATE_DEFINE_ENV_VAR(bool, LEGATE_INLINE_TASK_LAUNCH);

}  // namespace experimental

#undef LEGATE_DEFINE_ENV_VAR

}  // namespace legate::detail

#undef LEGATE_CHECK_ENV_VAR_DOCS
#define LEGATE_CHECK_ENV_VAR_DOCS(name) \
  static_assert(legate::detail::ZStringView{legate::detail::name} == #name)

#include <legate/utilities/env.h>
