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

static_assert(!is_pure_move_constructible_v<Legion::CodeDescriptor>,
              "Use by value and std::move for Legion::CodeDescriptor");

void TaskInfo::add_variant(VariantCode vid,
                           VariantImpl body,
                           Legion::CodeDescriptor&& code_desc,
                           const VariantOptions& options,
                           std::optional<InternalSharedPtr<TaskSignature>>
                             signature  // NOLINT(performance-unnecessary-value-param)
)
{
  // Legion::Runtime::has_context() is necessary here because it is possible we are
  // performing registration during Legion runtime initialization, in which each
  // rank will perform the registration locally for themselves, despite being in
  // single controller execution mode.
  if (Legion::Runtime::has_context() &&
      legate::detail::Runtime::get_runtime().config().single_controller_execution()) {
    if (!code_desc.create_portable_implementation()) {
      throw detail::TracedException<std::runtime_error>{fmt::format(
        "Single controller execution mode requires a portable implementation for task {}, but "
        "failed to create one. This is likely because the task's symbol is not visible. Ensure the "
        "task is not in an anonymous namespace and is templated with visible, named template "
        "parameters, if it has a template. Ensure the symbol is in the dynamic symbol table in "
        "your executable or shared library via `nm -D`.",
        name())};
    }
  }
  if (signature.has_value()) {
    (*signature)->validate(name().as_string_view());
  }
  if (!variants_().try_emplace(vid, body, code_desc, options, std::move(signature)).second) {
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

void TaskInfo::register_task(GlobalTaskID task_id) const
{
  auto* const runtime       = Legion::Runtime::get_runtime();
  const auto legion_task_id = static_cast<Legion::TaskID>(task_id);

  // Legion::Runtime::has_context() is necessary here because it is possible we are
  // performing registration during Legion runtime initialization, in which each
  // rank will perform the registration locally for themselves, despite being in
  // single controller execution mode.
  const bool global_registration =
    Legion::Runtime::has_context() &&
    legate::detail::Runtime::get_runtime().config().single_controller_execution();

  static_assert(std::is_same_v<std::decay_t<decltype(name())>, detail::ZStringView>);
  runtime->attach_name(legion_task_id,
                       name().data(),  // NOLINT(bugprone-suspicious-stringview-data-usage)
                       false /*mutable*/,
                       !global_registration /*local_only*/);
  for (auto&& [vcode, vinfo] : variants_()) {
    // variant_name is used only once (in the ctor of TaskVariantRegistrar) but Legion doesn't
    // actually strdup the string until runtime->register_task_variant(), so the string must
    // stay alive until that call.
    const auto variant_name = fmt::format("{}", vcode);
    auto registrar =
      Legion::TaskVariantRegistrar{
        legion_task_id, variant_name.c_str(), global_registration /*global*/}
        .add_constraint(Legion::ProcessorConstraint{mapping::detail::to_kind(vcode)});

    vinfo.options.populate_registrar(registrar);
    runtime->register_task_variant(registrar,
                                   vinfo.code_desc,
                                   /* user_data */ nullptr,
                                   /* user_len */ 0,
                                   /* return_size */ 0,
                                   to_underlying(vcode),
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
