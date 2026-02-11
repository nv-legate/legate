/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/task/task_info.h>

#include <legate/runtime/library.h>
#include <legate/task/detail/task_config.h>
#include <legate/task/detail/task_info.h>
#include <legate/task/detail/task_signature.h>
#include <legate/task/task_config.h>
#include <legate/task/task_signature.h>
#include <legate/utilities/typedefs.h>

#include <iostream>

namespace legate {

// This operation is not const, even if all member calls are.
// NOLINTNEXTLINE(readability-make-member-function-const)
void TaskInfo::add_variant_(AddVariantKey,
                            const Library& library,
                            VariantCode vid,
                            VariantImpl body,
                            Processor::TaskFuncPtr entry,
                            const VariantOptions* decl_options,
                            const std::map<VariantCode, VariantOptions>& registration_options)
{
  const auto& task_config = impl()->task_config();

  auto&& options = [&]() -> const VariantOptions& {
    // 1. The variant options (if any) supplied at the call-site of `register_variants()`.
    if (const auto it = registration_options.find(vid); it != registration_options.end()) {
      return it->second;
    }

    // 2. The default variant options (if any) found in `XXX_VARIANT_OPTIONS`.
    if (decl_options) {
      return *decl_options;
    }

    // 3. The variant options provided by TASK_CONFIG.
    if (const auto& task_options = task_config->variant_options(); task_options.has_value()) {
      return *task_options;
    }

    // 4. The variant options provided by `Library::get_default_variant_options()`.
    auto&& lib_defaults = library.get_default_variant_options();

    if (const auto it = lib_defaults.find(vid); it != lib_defaults.end()) {
      return it->second;
    }

    // 5. The global default variant options found in `VariantOptions::DEFAULT_OPTIONS`.
    return VariantOptions::DEFAULT_OPTIONS;
  }();

  std::optional<InternalSharedPtr<detail::TaskSignature>> signature;

  if (const auto& sig = task_config->signature(); sig.has_value()) {
    signature.emplace(*sig);
  }

  impl()->add_variant(vid, body, Legion::CodeDescriptor{entry}, options, std::move(signature));
}

// ==========================================================================================

TaskInfo::TaskInfo(std::string task_name, const TaskConfig& config)
  : impl_{legate::make_shared<detail::TaskInfo>(std::move(task_name), config.impl())}
{
}

std::string_view TaskInfo::name() const { return impl()->name().as_string_view(); }

std::optional<VariantInfo> TaskInfo::find_variant(VariantCode vid) const
{
  if (const auto priv = impl()->find_variant(vid)) {
    return VariantInfo{*priv};
  }
  return std::nullopt;
}

std::string TaskInfo::to_string() const { return impl()->to_string(); }

TaskConfig TaskInfo::task_config() const { return TaskConfig{impl()->task_config()}; }

std::ostream& operator<<(std::ostream& os, const TaskInfo& info)
{
  os << info.to_string();
  return os;
}

}  // namespace legate
