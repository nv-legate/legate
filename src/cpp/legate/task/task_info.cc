/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "legate/task/task_info.h"

#include "legate/utilities/detail/formatters.h"
#include "legate/utilities/detail/zstring_view.h"
#include "legate/utilities/typedefs.h"
#include <legate/utilities/detail/traced_exception.h>

#include <array>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <iostream>
#include <stdexcept>

namespace legate {

namespace {

constexpr std::array<detail::ZStringView, 4> VARIANT_NAMES = {"(invalid)", "CPU", "GPU", "OMP"};

constexpr std::array<Processor::Kind, 4> VARIANT_PROC_KINDS = {Processor::Kind::NO_KIND,
                                                               Processor::Kind::LOC_PROC,
                                                               Processor::Kind::TOC_PROC,
                                                               Processor::Kind::OMP_PROC};

}  // namespace

std::string VariantInfo::to_string() const
{
  return fmt::format("{:x}, {}", reinterpret_cast<std::uintptr_t>(body), fmt::streamed(options));
}

// ==========================================================================================

class TaskInfo::Impl {
 public:
  explicit Impl(std::string task_name);

  void add_variant(VariantCode vid,
                   VariantImpl body,
                   const Legion::CodeDescriptor& code_desc,
                   const VariantOptions& options);
  [[nodiscard]] std::optional<std::reference_wrapper<const VariantInfo>> find_variant(
    VariantCode vid) const;

  void register_task(GlobalTaskID task_id) const;

  [[nodiscard]] detail::ZStringView name() const;
  [[nodiscard]] const std::map<VariantCode, VariantInfo>& variants() const;

 private:
  std::string task_name_{};
  std::map<VariantCode, VariantInfo> variants_{};
};

TaskInfo::Impl::Impl(std::string task_name) : task_name_{std::move(task_name)} {}

static_assert(!traits::detail::is_pure_move_constructible_v<Legion::CodeDescriptor>,
              "Use by value and std::move for Legion::CodeDescriptor");
void TaskInfo::Impl::add_variant(VariantCode vid,
                                 VariantImpl body,
                                 const Legion::CodeDescriptor& code_desc,
                                 const VariantOptions& options)
{
  if (!variants_
         .emplace(std::piecewise_construct,
                  std::forward_as_tuple(vid),
                  std::forward_as_tuple(body, code_desc, options))
         .second) {
    throw detail::TracedException<std::invalid_argument>{
      fmt::format("Task {} already has variant {}", name(), vid)};
  }
}

std::optional<std::reference_wrapper<const VariantInfo>> TaskInfo::Impl::find_variant(
  VariantCode vid) const
{
  const auto it = variants().find(vid);

  if (it == variants().end()) {
    return std::nullopt;
  }
  return it->second;
}

void TaskInfo::Impl::register_task(GlobalTaskID task_id) const
{
  const auto runtime = Legion::Runtime::get_runtime();

  static_assert(std::is_same_v<std::decay_t<decltype(name())>, detail::ZStringView>);
  runtime->attach_name(static_cast<Legion::TaskID>(task_id),
                       name().data(),  // NOLINT(bugprone-suspicious-stringview-data-usage)
                       false /*mutable*/,
                       true /*local_only*/);
  for (auto&& [vcode, vinfo] : variants()) {
    const auto vid = traits::detail::to_underlying(vcode);
    auto&& options = vinfo.options;
    auto registrar = Legion::TaskVariantRegistrar{
      static_cast<Legion::TaskID>(task_id),
      false /*global*/,
      VARIANT_NAMES[vid].data()  // NOLINT(bugprone-suspicious-stringview-data-usage)
    };
    static_assert(std::is_same_v<std::decay_t<decltype(VARIANT_NAMES[vid])>, detail::ZStringView>);

    registrar.add_constraint(Legion::ProcessorConstraint{VARIANT_PROC_KINDS[vid]});
    options.populate_registrar(registrar);
    runtime->register_task_variant(registrar, vinfo.code_desc, nullptr, 0, 0, vid, false);
  }
}

detail::ZStringView TaskInfo::Impl::name() const { return task_name_; }

const std::map<VariantCode, VariantInfo>& TaskInfo::Impl::variants() const { return variants_; }

// ==========================================================================================

void TaskInfo::add_variant_(AddVariantKey,
                            Library library,
                            VariantCode vid,
                            VariantImpl body,
                            Processor::TaskFuncPtr entry,
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

  impl_->add_variant(vid, body, Legion::CodeDescriptor{entry}, options);
}

void TaskInfo::add_variant_(RuntimeAddVariantKey,
                            Library core_lib,
                            VariantCode vid,
                            const VariantOptions* callsite_options,
                            Legion::CodeDescriptor&& descr)
{
  auto&& options = [&]() -> const VariantOptions& {
    if (callsite_options) {
      return *callsite_options;
    }

    auto&& lib_defaults = core_lib.get_default_variant_options();
    const auto it       = lib_defaults.find(vid);

    return it == lib_defaults.end() ? VariantOptions::DEFAULT_OPTIONS : it->second;
  }();

  impl_->add_variant(vid, nullptr, descr, options);
}

// ==========================================================================================

TaskInfo::TaskInfo(std::string task_name) : impl_{std::make_unique<Impl>(std::move(task_name))} {}

TaskInfo::~TaskInfo() = default;

std::string_view TaskInfo::name() const { return impl_->name().as_string_view(); }

std::optional<std::reference_wrapper<const VariantInfo>> TaskInfo::find_variant(
  VariantCode vid) const
{
  return impl_->find_variant(vid);
}

bool TaskInfo::has_variant(VariantCode vid) const { return find_variant(vid).has_value(); }

void TaskInfo::register_task(GlobalTaskID task_id) { impl_->register_task(task_id); }

std::string TaskInfo::to_string() const
{
  std::string ret;

  fmt::format_to(std::back_inserter(ret), "{} {{", name());
  for (auto&& [vid, vinfo] : impl_->variants()) {
    fmt::format_to(std::back_inserter(ret), "{}:[{}],", vid, vinfo);
  }
  ret += '}';
  return ret;
}

std::ostream& operator<<(std::ostream& os, const TaskInfo& info)
{
  os << info.to_string();
  return os;
}

}  // namespace legate
