/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/task/task_config.h>

#include <legate/task/detail/task_config.h>
#include <legate/task/task_signature.h>
#include <legate/utilities/shared_ptr.h>

#include <functional>
#include <optional>

namespace legate {

TaskConfig::TaskConfig(LocalTaskID task_id)
  : pimpl_{legate::make_shared<detail::TaskConfig>(task_id)}
{
}

TaskConfig& TaskConfig::with_signature(const TaskSignature& signature)
{
  impl()->signature(signature.impl());
  return *this;
}

TaskConfig& TaskConfig::with_variant_options(const VariantOptions& options)
{
  impl()->variant_options(options);
  return *this;
}

LocalTaskID TaskConfig::task_id() const { return impl()->task_id(); }

std::optional<TaskSignature> TaskConfig::task_signature() const
{
  if (const auto& sig = impl()->signature(); sig.has_value()) {
    return {TaskSignature{*sig}};
  }
  return std::nullopt;
}

std::optional<std::reference_wrapper<const VariantOptions>> TaskConfig::variant_options() const
{
  if (const auto& options = impl()->variant_options(); options.has_value()) {
    return {std::cref(*options)};
  }
  return std::nullopt;
}

bool operator==(const TaskConfig& lhs, const TaskConfig& rhs) noexcept
{
  return *lhs.impl() == *rhs.impl();
}

bool operator!=(const TaskConfig& lhs, const TaskConfig& rhs) noexcept
{
  return *lhs.impl() != *rhs.impl();
}

TaskConfig::~TaskConfig() = default;

}  // namespace legate
