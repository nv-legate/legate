/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "core/utilities/macros.h"

#include <cstdint>
#include <optional>
#include <string_view>

/** @addtogroup util
 *  @{
 */

/**
 * @file
 * @brief Definitions of global environment variables which are understood by Legate.
 */

namespace legate {

namespace detail {

class EnvironmentVariableBase {
 public:
  EnvironmentVariableBase() = delete;
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr EnvironmentVariableBase(std::string_view name) noexcept;

  // NOLINTNEXTLINE(google-explicit-constructor)
  [[nodiscard]] constexpr operator std::string_view() const noexcept;
  [[nodiscard]] constexpr const char* data() const noexcept;
  void set(std::string_view value, bool overwrite) const noexcept;

 private:
  std::string_view name_{};
};

// Need to define these here (instead of a .inl) so that they can be used in constexpr contexts
// below
constexpr EnvironmentVariableBase::EnvironmentVariableBase(std::string_view name) noexcept
  : name_{std::move(name)}
{
}

constexpr EnvironmentVariableBase::operator std::string_view() const noexcept { return name_; }

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

}  // namespace detail

#define LEGATE_ENV_BOOL_CONSTANT(NAME) \
  inline constexpr legate::detail::EnvironmentVariable<bool> NAME { LegateStringize(NAME) }

#define LEGATE_ENV_INT64_CONSTANT(NAME) \
  inline constexpr legate::detail::EnvironmentVariable<std::int64_t> NAME { LegateStringize(NAME) }

#define LEGATE_ENV_UINT32_CONSTANT(NAME) \
  inline constexpr legate::detail::EnvironmentVariable<std::uint32_t> NAME { LegateStringize(NAME) }

#define LEGATE_ENV_CONSTANT(NAME) inline constexpr std::string_view NAME = LegateStringize(NAME)

/**
 * @var LEGATE_TEST
 *
 * @brief Enables "testing" mode in Legate. Possible values: 0, 1.
 */
LEGATE_ENV_BOOL_CONSTANT(LEGATE_TEST);

/**
 * @var LEGATE_SHOW_USAGE
 *
 * @brief Enables verbose resource consumption logging of the base mapper on
 * desctruction. Possible values: 0, 1.
 */
LEGATE_ENV_BOOL_CONSTANT(LEGATE_SHOW_USAGE);

/**
 * @var LEGATE_NEED_CUDA
 *
 * @brief Instructs Legate that it must be CUDA-aware. Possible values: 0, 1.
 *
 * Enabling this, means that Legate must have been configured with CUDA support, and that a
 * CUDA-capable device must be present at startup. If either of these conditions are not met,
 * Legate will abort execution.
 */
LEGATE_ENV_BOOL_CONSTANT(LEGATE_NEED_CUDA);

/**
 * @var LEGATE_NEED_OPENMP
 *
 * @brief Instructs Legate that it must be OpenMP-aware. Possible values: 0, 1.
 *
 * Enabling this, means that Legate must have been configured with OpenMP support, and that a
 * OpenMP-capable device must be present at startup. If either of these conditions are not met,
 * Legate will abort execution.
 */
LEGATE_ENV_BOOL_CONSTANT(LEGATE_NEED_OPENMP);

/**
 * @var LEGATE_NEED_NETWORK
 *
 * @brief Instructs Legate that it must be network-aware. Possible values: 0, 1
 *
 * Enabling this, means that Legate must have been configured with networking support. If
 * either of this condition is not met, Legate will abort execution.
 */
LEGATE_ENV_BOOL_CONSTANT(LEGATE_NEED_NETWORK);

/**
 * @var LEGATE_SHOW_PROGRESS
 *
 * @brief Instructs Legate to emit basic info at that start of each task. Possible values: 0,
 * 1.
 *
 * This variable is useful to visually ensure that a particular task is being called. The
 * progress reports are emitted by Legate before entering into the task body itself.
 */
LEGATE_ENV_BOOL_CONSTANT(LEGATE_SHOW_PROGRESS);

/**
 * @var LEGATE_EMPTY_TASK
 *
 * @brief Instructs Legate to use a dummy empty task body for each task. Possible values: 0, 1.
 *
 * This variable may be enabled to debug logical issues between tasks (for example, control
 * replication issues) by executing the entire task graph without needing to execute the task
 * bodies themselves. This is particularly useful if the task bodies are expensive.
 */
LEGATE_ENV_BOOL_CONSTANT(LEGATE_EMPTY_TASK);

/**
 * @var LEGATE_SYNC_STREAM_VIEW
 *
 * @brief Instructs Legate to synchronize CUDA streams before destruction. Possible values: 0, 1.
 *
 * This variable may be enabled to debug asynchronous issues with CUDA streams. A program which
 * produces different results with this variable enabled and disabled very likely has a race
 * condition between streams. This is especially useful when combined with
 * CUDA_LAUNCH_BLOCKING.
 */
LEGATE_ENV_BOOL_CONSTANT(LEGATE_SYNC_STREAM_VIEW);

/**
 * @var LEGATE_LOG_MAPPING
 *
 * @brief Instructs Legate to emit mapping decisions to stdout. Possible values: 0, 1.
 */
LEGATE_ENV_BOOL_CONSTANT(LEGATE_LOG_MAPPING);

/**
 * @var LEGATE_LOG_MAPPING
 *
 * @brief Instructs Legate to emit partitioning decisions to stdout. Possible values: 0, 1.
 */
LEGATE_ENV_BOOL_CONSTANT(LEGATE_LOG_PARTITIONING);

/**
 * @var LEGATE_WARMUP_NCCL
 *
 * @brief Instructs Legate to "warm up" NCCL during startup. Possible values: 0, 1.
 *
 * NCCL usually has a relatively high startup cost the first time any collective communication
 * is performed. This could corrupt performance measurements if that startup is performed in
 * the hot-path.
 */
LEGATE_ENV_BOOL_CONSTANT(LEGATE_WARMUP_NCCL);

/**
 * @var LEGION_DEFAULT_ARGS
 *
 * @brief Default arguments to pass to Legion initialization. Possible values: a string.
 *
 * These arguments are passed verbatim to Legion during runtime startup.
 */
LEGATE_ENV_CONSTANT(LEGION_DEFAULT_ARGS);

LEGATE_ENV_INT64_CONSTANT(LEGATE_MIN_CPU_CHUNK);
LEGATE_ENV_INT64_CONSTANT(LEGATE_MIN_GPU_CHUNK);
LEGATE_ENV_INT64_CONSTANT(LEGATE_MIN_OMP_CHUNK);
LEGATE_ENV_UINT32_CONSTANT(LEGATE_WINDOW_SIZE);
LEGATE_ENV_UINT32_CONSTANT(LEGATE_FIELD_REUSE_FRAC);
LEGATE_ENV_UINT32_CONSTANT(LEGATE_FIELD_REUSE_FREQ);
LEGATE_ENV_UINT32_CONSTANT(LEGATE_MAX_LRU_LENGTH);
LEGATE_ENV_BOOL_CONSTANT(LEGATE_CONSENSUS);
LEGATE_ENV_BOOL_CONSTANT(LEGATE_DISABLE_MPI);

#undef LEGATE_ENV_CONSTANT
#undef LEGATE_ENV_BOOL_CONSTANT
#undef LEGATE_ENV_UINT32_CONSTANT
#undef LEGATE_ENV_INT64_CONSTANT

}  // namespace legate

/** @} */  // end of group util
