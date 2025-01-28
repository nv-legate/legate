/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <legate/task/detail/task_info.h>

#include <legate/mapping/detail/mapping.h>
#include <legate/runtime/detail/library.h>
#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/detail/type_traits.h>

#include <fmt/format.h>

#include <stdexcept>
#include <tuple>
#include <type_traits>

namespace legate::detail {

static_assert(!traits::detail::is_pure_move_constructible_v<Legion::CodeDescriptor>,
              "Use by value and std::move for Legion::CodeDescriptor");
void TaskInfo::add_variant(VariantCode vid,
                           VariantImpl body,
                           const Legion::CodeDescriptor& code_desc,
                           const VariantOptions& options)
{
  if (!variants_().try_emplace(vid, body, code_desc, options).second) {
    throw detail::TracedException<std::invalid_argument>{
      fmt::format("Task {} already has variant {}", name(), vid)};
  }
}

void TaskInfo::add_variant_(RuntimeAddVariantKey,
                            const Library& core_lib,
                            VariantCode vid,
                            const VariantOptions* callsite_options,
                            const Legion::CodeDescriptor& descr)
{
  auto&& options = [&]() -> const VariantOptions& {
    if (callsite_options) {
      return *callsite_options;
    }

    auto&& lib_defaults = core_lib.get_default_variant_options();
    const auto it       = lib_defaults.find(vid);

    return it == lib_defaults.end() ? VariantOptions::DEFAULT_OPTIONS : it->second;
  }();

  add_variant(vid, nullptr, descr, options);
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

void TaskInfo::register_task(GlobalTaskID task_id) const
{
  auto* const runtime       = Legion::Runtime::get_runtime();
  const auto legion_task_id = static_cast<Legion::TaskID>(task_id);

  static_assert(std::is_same_v<std::decay_t<decltype(name())>, detail::ZStringView>);
  runtime->attach_name(legion_task_id,
                       name().data(),  // NOLINT(bugprone-suspicious-stringview-data-usage)
                       false /*mutable*/,
                       true /*local_only*/);
  for (auto&& [vcode, vinfo] : variants_()) {
    // variant_name is used only once (in the ctor of TaskVariantRegistrar) but Legion doesn't
    // actually strdup the string until runtime->register_task_variant(), so the string must
    // stay alive until that call.
    const auto variant_name = fmt::format("{}", vcode);
    auto registrar =
      Legion::TaskVariantRegistrar{legion_task_id, variant_name.c_str(), false /*global*/}
        .add_constraint(Legion::ProcessorConstraint{mapping::detail::to_kind(vcode)});

    vinfo.options.populate_registrar(registrar);
    runtime->register_task_variant(registrar,
                                   vinfo.code_desc,
                                   /* user_data */ nullptr,
                                   /* user_len */ 0,
                                   /* return_size */ 0,
                                   traits::detail::to_underlying(vcode),
                                   /* has_return_type_size */ false);
  }
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
