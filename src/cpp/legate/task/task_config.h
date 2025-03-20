/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/task/task_signature.h>
#include <legate/utilities/detail/doxygen.h>
#include <legate/utilities/shared_ptr.h>
#include <legate/utilities/typedefs.h>

#include <functional>
#include <optional>

/** @file */

namespace legate::detail {

class TaskConfig;

}  // namespace legate::detail

namespace legate {

class VariantOptions;

/**
 * @addtogroup task
 * @{
 */

/**
 * @brief A class representing the configuration of a task.
 *
 * This class provides methods for constructing a task configuration, setting various task
 * options, and retrieving information about the task configuration.
 */
class TaskConfig {
 public:
  /**
   * @brief Deleted default constructor.
   *
   * The default constructor is deleted to prevent creating a `TaskConfig` object without
   * specifying a task ID.
   */
  TaskConfig()                                 = LEGATE_DEFAULT_WHEN_CYTHON;
  TaskConfig(const TaskConfig&)                = default;
  TaskConfig& operator=(const TaskConfig&)     = default;
  TaskConfig(TaskConfig&&) noexcept            = default;
  TaskConfig& operator=(TaskConfig&&) noexcept = default;
  ~TaskConfig();

  /**
   * @brief Construct a TaskConfig.
   *
   * @param task_id The local ID of the task.
   */
  explicit TaskConfig(LocalTaskID task_id);

  /**
   * @brief Set the task signature for this task.
   *
   * @param signature The task signature to associate with the task.
   * @return A reference to `this`.
   */
  TaskConfig& with_signature(const TaskSignature& signature);

  /**
   * @brief Set the variant options for this task.
   *
   * @param options The variant options to associate with the task.
   * @return A reference to `this`.
   */
  TaskConfig& with_variant_options(const VariantOptions& options);

  /**
   * @return The local task ID for this task.
   */
  [[nodiscard]] LocalTaskID task_id() const;

  /**
   * @return The task signature, if set, `std::nullopt` otherwise.
   */
  [[nodiscard]] std::optional<TaskSignature> task_signature() const;

  /**
   * @return The variant options, if set, `std::nullopt` otherwise.
   */
  [[nodiscard]] std::optional<std::reference_wrapper<const VariantOptions>> variant_options() const;

  [[nodiscard]] const SharedPtr<detail::TaskConfig>& impl() const;

  friend bool operator==(const TaskConfig& lhs, const TaskConfig& rhs) noexcept;
  friend bool operator!=(const TaskConfig& lhs, const TaskConfig& rhs) noexcept;

 private:
  SharedPtr<detail::TaskConfig> pimpl_{};
};

/** @} */

}  // namespace legate

#include <legate/task/task_config.inl>
