/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/task/detail/task_info.h>

#include <legate/mapping/detail/mapping.h>
#include <legate/runtime/detail/library.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/detail/type_traits.h>

#include <fmt/format.h>

#include <stdexcept>
#include <tuple>
#include <type_traits>

namespace legate::detail {

TaskInfo::TaskInfo(std::string task_name, InternalSharedPtr<TaskConfig> task_config)
  : task_name_{std::move(task_name)}, task_config_{std::move(task_config)}
{
}

static_assert(!is_pure_move_constructible_v<Legion::CodeDescriptor>,
              "Use by value and std::move for Legion::CodeDescriptor");

void TaskInfo::add_variant(VariantCode vid,
                           VariantImpl body,
                           const VariantOptions& options,
                           std::optional<InternalSharedPtr<TaskSignature>>
                             signature  // NOLINT(performance-unnecessary-value-param)
)
{
  if (signature.has_value()) {
    (*signature)->validate(name().as_string_view());
  }
  if (!variants_().try_emplace(vid, body, options, std::move(signature)).second) {
    throw detail::TracedException<std::invalid_argument>{
      fmt::format("Task {} already has variant {}", name(), vid)};
  }
}

std::optional<std::reference_wrapper<const VariantInfo>> TaskInfo::find_variant(
  VariantCode vid) const
{
  const auto it = variants_().find(vid);

  if (it == variants_().end()) {
    return std::nullopt;
  }
  return it->second;
}

std::string TaskInfo::to_string() const
{
  std::string ret;

  fmt::format_to(std::back_inserter(ret), "{} {{", name());
  for (auto&& [vid, vinfo] : variants_()) {
    fmt::format_to(std::back_inserter(ret), "{}:[{}],", vid, vinfo);
  }
  ret += '}';
  return ret;
}

}  // namespace legate::detail
