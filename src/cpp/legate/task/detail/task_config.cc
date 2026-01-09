/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/task/detail/task_config.h>

#include <legate/task/detail/task_signature.h>  // for operator==

namespace legate::detail {

TaskConfig::TaskConfig(LocalTaskID task_id) : task_id_{task_id} {}

void TaskConfig::signature(InternalSharedPtr<TaskSignature> signature)
{
  signature_ = std::move(signature);
}

void TaskConfig::variant_options(const VariantOptions& options) { variant_options_ = options; }

bool operator==(const TaskConfig& lhs, const TaskConfig& rhs) noexcept
{
  if (std::addressof(lhs) == std::addressof(rhs)) {
    return true;
  }

  if (lhs.task_id() != rhs.task_id()) {
    return false;
  }

  if (lhs.variant_options() != rhs.variant_options()) {
    return false;
  }

  const auto& lhs_sig = lhs.signature();
  const auto& rhs_sig = rhs.signature();

  if (lhs_sig.has_value() && rhs_sig.has_value()) {
    return (**lhs_sig) == (**rhs_sig);
  }
  return !lhs_sig.has_value() && !rhs_sig.has_value();
}

bool operator!=(const TaskConfig& lhs, const TaskConfig& rhs) noexcept { return !(lhs == rhs); }

}  // namespace legate::detail
