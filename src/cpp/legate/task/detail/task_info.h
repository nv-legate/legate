/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/task/detail/variant_info.h>
#include <legate/task/variant_options.h>
#include <legate/utilities/detail/zstring_view.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/typedefs.h>

#include <legion/api/types.h>

#include <functional>
#include <map>
#include <optional>
#include <string>

namespace legate::detail {

class TaskSignature;

/**
 * @brief A class holding the task registration info.
 */
class TaskInfo {
 public:
  /**
   * @brief Construct a TaskInfo
   *
   * @param task_name The name of the task, usually gotten via demangling std::type_info.
   */
  explicit TaskInfo(std::string task_name);

  /**
   * @brief Append a new variant to the task.
   *
   * @param vid The variant kind to add.
   * @param body The variant task body.
   * @param code_desc The Legion code descriptor to be passed on to Legion.
   * @param options The variant options.
   */
  void add_variant(VariantCode vid,
                   VariantImpl body,
                   Legion::CodeDescriptor&& code_desc,
                   const VariantOptions& options,
                   std::optional<InternalSharedPtr<TaskSignature>> signature);

  /**
   * @brief Find a particular variant for the task.
   *
   * @param vid The variant to find.
   *
   * @return An optional containing the variant, if it is found, `std::nullopt` otherwise.
   *
   * You can implement `has_variant()` by simply calling `find_variant().has_value()`.
   */
  [[nodiscard]] std::optional<std::reference_wrapper<const VariantInfo>> find_variant(
    VariantCode vid) const;

  /**
   * @brief Register the task with Legion.
   *
   * @param task_id The global Legion task ID to register the task with.
   *
   * After this call,
   */
  void register_task(GlobalTaskID task_id) const;

  /**
   * @return A human-readable representation of the Task.
   */
  [[nodiscard]] std::string to_string() const;

  /**
   * @return The name of the task, usually the demangled name of the function.
   */
  [[nodiscard]] detail::ZStringView name() const;

 private:
  /**
   * @return The set of registered variants.
   */
  [[nodiscard]] const std::map<VariantCode, VariantInfo>& variants_() const;

  /**
   * @return The set of registered variants.
   */
  [[nodiscard]] std::map<VariantCode, VariantInfo>& variants_();

  std::string task_name_{};
  std::map<VariantCode, VariantInfo> task_variants_{};
};

}  // namespace legate::detail

#include <legate/task/detail/task_info.inl>
