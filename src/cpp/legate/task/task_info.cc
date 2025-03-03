/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/task/task_info.h>

#include <legate/runtime/library.h>
#include <legate/task/detail/task_info.h>
#include <legate/task/detail/task_signature.h>
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
                            const TaskSignature* signature,
                            const VariantOptions* decl_options,
                            const std::map<VariantCode, VariantOptions>& registration_options)
{
  auto&& options = [&]() -> const VariantOptions& {
    // 1. The variant options (if any) supplied at the call-site of `register_variants()`.
    if (const auto it = registration_options.find(vid); it != registration_options.end()) {
      return it->second;
    }

    // 2. The default variant options (if any) found in `XXX_VARIANT_OPTIONS`.
    if (decl_options) {
      return *decl_options;
    }

    // 3. The variant options provided by `Library::get_default_variant_options()`.
    auto&& lib_defaults = library.get_default_variant_options();

    if (const auto it = lib_defaults.find(vid); it != lib_defaults.end()) {
      return it->second;
    }

    // 4. The global default variant options found in `VariantOptions::DEFAULT_OPTIONS`.
    return VariantOptions::DEFAULT_OPTIONS;
  }();

  std::optional<InternalSharedPtr<detail::TaskSignature>> sig;

  if (signature) {
    sig.emplace(signature->impl());
  }

  impl()->add_variant(vid, body, Legion::CodeDescriptor{entry}, options, std::move(sig));
}

// ==========================================================================================

TaskInfo::TaskInfo(std::string task_name)
  : impl_{legate::make_shared<detail::TaskInfo>(std::move(task_name))}
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

std::ostream& operator<<(std::ostream& os, const TaskInfo& info)
{
  os << info.to_string();
  return os;
}

}  // namespace legate
