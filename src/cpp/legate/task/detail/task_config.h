/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/task/detail/store_mapping_signature.h>
#include <legate/task/variant_options.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/typedefs.h>

#include <optional>

namespace legate::detail {

class TaskSignature;

/**
 * @brief Private implementation of the task configuration parameters.
 */
class TaskConfig {
 public:
  /**
   * @brief TaskConfig cannot be default constructed. The user must declare the task ID.
   */
  TaskConfig() = delete;

  /**
   * @brief Construct a TaskConfig.
   *
   * @param task_id The local ID of the task.
   */
  explicit TaskConfig(LocalTaskID task_id);

  /**
   * @brief Register the task signature.
   *
   * This overrides the currently stored task signature. There is currently no way to clear the
   * currently registered set of options.
   *
   * @param signature The task signature to register.
   */
  void signature(InternalSharedPtr<TaskSignature> signature);

  /**
   * @brief Register task-wide variant options.
   *
   * If the user does not set variant-specific options, these will serve as the
   * default. Otherwise, defaults specified in the library will be chosen.
   *
   * @param options The variant options to register.
   */
  void variant_options(const VariantOptions& options);

  /**
   * @brief Register task-wide mapping options.
   *
   * @param store_mappings The store mapping signature.
   */
  void store_mappings(StoreMappingSignature store_mappings);

  /**
   * @return The local task ID for this task.
   */
  [[nodiscard]] LocalTaskID task_id() const;

  /**
   * @return The task signature for the task or `std::nullopt` if no signature was set.
   */
  [[nodiscard]] const std::optional<InternalSharedPtr<TaskSignature>>& signature() const;

  /**
   * @return The task-wide variant options for the task, or `std::nullopt` if no variant
   * options were set.
   */
  [[nodiscard]] const std::optional<VariantOptions>& variant_options() const;

  /**
   * @return The task-wide mapping options for the task, or `std::nullopt` if none were set.
   */
  [[nodiscard]] const std::optional<StoreMappingSignature>& store_mappings() const;

  friend bool operator==(const TaskConfig& lhs, const TaskConfig& rhs) noexcept;
  friend bool operator!=(const TaskConfig& lhs, const TaskConfig& rhs) noexcept;

 private:
  LocalTaskID task_id_{};
  std::optional<InternalSharedPtr<TaskSignature>> signature_{};
  std::optional<VariantOptions> variant_options_{};
  std::optional<StoreMappingSignature> store_mappings_{};
};

}  // namespace legate::detail

#include <legate/task/detail/task_config.inl>
