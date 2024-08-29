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

#pragma once

#include "core/utilities/detail/zstring_view.h"

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

inline constexpr EnvironmentVariable<bool> LEGATE_TEST{"LEGATE_TEST"};
inline constexpr EnvironmentVariable<bool> LEGATE_SHOW_USAGE{"LEGATE_SHOW_USAGE"};
inline constexpr EnvironmentVariable<bool> LEGATE_NEED_CUDA{"LEGATE_NEED_CUDA"};
inline constexpr EnvironmentVariable<bool> LEGATE_NEED_OPENMP{"LEGATE_NEED_OPENMP"};
inline constexpr EnvironmentVariable<bool> LEGATE_NEED_NETWORK{"LEGATE_NEED_NETWORK"};
inline constexpr EnvironmentVariable<bool> LEGATE_SHOW_PROGRESS{"LEGATE_SHOW_PROGRESS"};
inline constexpr EnvironmentVariable<bool> LEGATE_EMPTY_TASK{"LEGATE_EMPTY_TASK"};
inline constexpr EnvironmentVariable<bool> LEGATE_SYNC_STREAM_VIEW{"LEGATE_SYNC_STREAM_VIEW"};
inline constexpr EnvironmentVariable<bool> LEGATE_LOG_MAPPING{"LEGATE_LOG_MAPPING"};
inline constexpr EnvironmentVariable<bool> LEGATE_LOG_PARTITIONING{"LEGATE_LOG_PARTITIONING"};
inline constexpr EnvironmentVariable<bool> LEGATE_WARMUP_NCCL{"LEGATE_WARMUP_NCCL"};
inline constexpr EnvironmentVariable<std::string> LEGION_DEFAULT_ARGS{"LEGION_DEFAULT_ARGS"};
inline constexpr EnvironmentVariable<std::int64_t> LEGATE_MIN_CPU_CHUNK{"LEGATE_MIN_CPU_CHUNK"};
inline constexpr EnvironmentVariable<std::int64_t> LEGATE_MIN_GPU_CHUNK{"LEGATE_MIN_GPU_CHUNK"};
inline constexpr EnvironmentVariable<std::int64_t> LEGATE_MIN_OMP_CHUNK{"LEGATE_MIN_OMP_CHUNK"};
inline constexpr EnvironmentVariable<std::uint32_t> LEGATE_WINDOW_SIZE{"LEGATE_WINDOW_SIZE"};
inline constexpr EnvironmentVariable<std::uint32_t> LEGATE_FIELD_REUSE_FRAC{
  "LEGATE_FIELD_REUSE_FRAC"};
inline constexpr EnvironmentVariable<std::uint32_t> LEGATE_FIELD_REUSE_FREQ{
  "LEGATE_FIELD_REUSE_FREQ"};
inline constexpr EnvironmentVariable<bool> LEGATE_CONSENSUS{"LEGATE_CONSENSUS"};
inline constexpr EnvironmentVariable<bool> LEGATE_DISABLE_MPI{"LEGATE_DISABLE_MPI"};
inline constexpr EnvironmentVariable<std::string> LEGATE_CONFIG{"LEGATE_CONFIG"};
inline constexpr EnvironmentVariable<std::string> LEGATE_MPI_WRAPPER{"LEGATE_MPI_WRAPPER"};

inline namespace experimental {

inline constexpr detail::EnvironmentVariable<bool> LEGATE_INLINE_TASK_LAUNCH{
  "LEGATE_INLINE_TASK_LAUNCH"};

}  // namespace experimental

}  // namespace legate::detail

#undef LEGATE_CHECK_ENV_VAR_DOCS
#define LEGATE_CHECK_ENV_VAR_DOCS(name) \
  static_assert(legate::detail::ZStringView{legate::detail::name} == #name)

#include "core/utilities/env.h"
