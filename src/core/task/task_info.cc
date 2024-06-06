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

#include "core/task/task_info.h"

#include <array>
#include <sstream>

namespace legate {

namespace {

constexpr std::array<std::string_view, 4> VARIANT_NAMES = {"(invalid)", "CPU", "GPU", "OMP"};

constexpr std::array<Processor::Kind, 4> VARIANT_PROC_KINDS = {Processor::Kind::NO_KIND,
                                                               Processor::Kind::LOC_PROC,
                                                               Processor::Kind::TOC_PROC,
                                                               Processor::Kind::OMP_PROC};

}  // namespace

class TaskInfo::Impl {
 public:
  explicit Impl(std::string task_name);

  void add_variant(LegateVariantCode vid,
                   VariantImpl body,
                   const Legion::CodeDescriptor& code_desc,
                   const VariantOptions& options);
  [[nodiscard]] const VariantInfo& find_variant(LegateVariantCode vid) const;
  [[nodiscard]] bool has_variant(LegateVariantCode vid) const;

  void register_task(Legion::TaskID task_id) const;

  [[nodiscard]] std::string_view name() const;
  [[nodiscard]] const std::map<LegateVariantCode, VariantInfo>& variants() const;

 private:
  std::string task_name_{};
  std::map<LegateVariantCode, VariantInfo> variants_{};
};

TaskInfo::Impl::Impl(std::string task_name) : task_name_{std::move(task_name)} {}

static_assert(!traits::detail::is_pure_move_constructible_v<Legion::CodeDescriptor>,
              "Use by value and std::move for Legion::CodeDescriptor");
void TaskInfo::Impl::add_variant(LegateVariantCode vid,
                                 VariantImpl body,
                                 const Legion::CodeDescriptor& code_desc,
                                 const VariantOptions& options)
{
  if (!variants_
         .emplace(std::piecewise_construct,
                  std::forward_as_tuple(vid),
                  std::forward_as_tuple(body, code_desc, options))
         .second) {
    std::stringstream ss;

    ss << "Task " << name() << " already has variant " << vid;
    throw std::invalid_argument{std::move(ss).str()};
  }
}

const VariantInfo& TaskInfo::Impl::find_variant(LegateVariantCode vid) const
{
  return variants().at(vid);
}

bool TaskInfo::Impl::has_variant(LegateVariantCode vid) const
{
  return variants().find(vid) != variants().end();
}

void TaskInfo::Impl::register_task(Legion::TaskID task_id) const
{
  const auto runtime = Legion::Runtime::get_runtime();

  runtime->attach_name(task_id, name().data(), false /*mutable*/, true /*local_only*/);
  for (auto&& [vid, vinfo] : variants()) {
    auto&& options = vinfo.options;
    Legion::TaskVariantRegistrar registrar{task_id, false /*global*/, VARIANT_NAMES[vid].data()};

    registrar.add_constraint(Legion::ProcessorConstraint{VARIANT_PROC_KINDS[vid]});
    options.populate_registrar(registrar);
    runtime->register_task_variant(
      registrar, vinfo.code_desc, nullptr, 0, options.return_size, vid);
  }
}

std::string_view TaskInfo::Impl::name() const { return task_name_; }

const std::map<LegateVariantCode, VariantInfo>& TaskInfo::Impl::variants() const
{
  return variants_;
}

// ==========================================================================================

TaskInfo::TaskInfo(std::string task_name) : impl_{std::make_unique<Impl>(std::move(task_name))} {}

TaskInfo::~TaskInfo() = default;

std::string_view TaskInfo::name() const { return impl_->name(); }

static_assert(!traits::detail::is_pure_move_constructible_v<Legion::CodeDescriptor>,
              "Use by value and std::move for Legion::CodeDescriptor");
void TaskInfo::add_variant(LegateVariantCode vid,
                           VariantImpl body,
                           const Legion::CodeDescriptor& code_desc,
                           const VariantOptions& options)
{
  impl_->add_variant(vid, body, code_desc, options);
}

void TaskInfo::add_variant(LegateVariantCode vid,
                           VariantImpl body,
                           RealmCallbackFn entry,
                           const VariantOptions& default_options,
                           const std::map<LegateVariantCode, VariantOptions>& all_options)
{
  const auto finder = all_options.find(vid);

  add_variant(vid,
              body,
              Legion::CodeDescriptor{entry},
              finder == all_options.end() ? default_options : finder->second);
}

void TaskInfo::add_variant(LegateVariantCode vid,
                           VariantImpl body,
                           RealmCallbackFn entry,
                           const std::map<LegateVariantCode, VariantOptions>& all_options)
{
  add_variant(vid, body, entry, VariantOptions::DEFAULT_OPTIONS, all_options);
}

const VariantInfo& TaskInfo::find_variant(LegateVariantCode vid) const
{
  return impl_->find_variant(vid);
}

bool TaskInfo::has_variant(LegateVariantCode vid) const { return impl_->has_variant(vid); }

void TaskInfo::register_task(Legion::TaskID task_id) { impl_->register_task(task_id); }

std::ostream& operator<<(std::ostream& os, const VariantInfo& info)
{
  std::stringstream ss;

  // use ss instead of piping directly to ostream because showbase and hex are permanent
  // modifiers.
  ss << std::showbase << std::hex << reinterpret_cast<std::uintptr_t>(info.body) << ","
     << info.options;
  os << std::move(ss).str();
  return os;
}

std::ostream& operator<<(std::ostream& os, const TaskInfo& info)
{
  os << info.name() << " {";
  for (auto&& [vid, vinfo] : info.impl_->variants()) {
    os << VARIANT_NAMES[vid] << ":[" << vinfo << "],";
  }
  os << "}";
  return os;
}

}  // namespace legate
