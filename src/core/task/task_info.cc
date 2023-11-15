/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace legate {

namespace {

const char* VARIANT_NAMES[] = {"(invalid)", "CPU", "GPU", "OMP"};

const Processor::Kind VARIANT_PROC_KINDS[] = {
  Processor::Kind::NO_KIND,
  Processor::Kind::LOC_PROC,
  Processor::Kind::TOC_PROC,
  Processor::Kind::OMP_PROC,
};

}  // namespace

class TaskInfo::Impl {
 public:
  Impl(std::string task_name);

  [[nodiscard]] const std::string& name() const { return task_name_; }

  void add_variant(LegateVariantCode vid,
                   VariantImpl body,
                   const Legion::CodeDescriptor& code_desc,
                   const VariantOptions& options);
  [[nodiscard]] const VariantInfo& find_variant(LegateVariantCode vid) const;
  [[nodiscard]] bool has_variant(LegateVariantCode vid) const;

  void register_task(Legion::TaskID task_id);

  [[nodiscard]] const std::map<LegateVariantCode, VariantInfo>& variants() const
  {
    return variants_;
  }

 private:
  std::string task_name_;
  std::map<LegateVariantCode, VariantInfo> variants_{};
};

TaskInfo::Impl::Impl(std::string task_name) : task_name_{std::move(task_name)} {}

const std::string& TaskInfo::name() const { return impl_->name(); }

void TaskInfo::Impl::add_variant(LegateVariantCode vid,
                                 VariantImpl body,
                                 const Legion::CodeDescriptor& code_desc,
                                 const VariantOptions& options)
{
  if (variants_.find(vid) != variants_.end()) {
    throw std::invalid_argument("Task " + task_name_ + " already has variant " +
                                std::to_string(vid));
  }
  variants_.emplace(vid, VariantInfo{body, code_desc, options});
}

const VariantInfo& TaskInfo::Impl::find_variant(LegateVariantCode vid) const
{
  return variants_.at(vid);
}

bool TaskInfo::Impl::has_variant(LegateVariantCode vid) const
{
  return variants_.find(vid) != variants_.end();
}

void TaskInfo::Impl::register_task(Legion::TaskID task_id)
{
  auto runtime = Legion::Runtime::get_runtime();
  runtime->attach_name(task_id, task_name_.c_str(), false /*mutable*/, true /*local_only*/);
  for (auto& [vid, vinfo] : variants_) {
    Legion::TaskVariantRegistrar registrar(task_id, false /*global*/, VARIANT_NAMES[vid]);
    registrar.add_constraint(Legion::ProcessorConstraint(VARIANT_PROC_KINDS[vid]));
    vinfo.options.populate_registrar(registrar);
    runtime->register_task_variant(
      registrar, vinfo.code_desc, nullptr, 0, vinfo.options.return_size, vid);
  }
}

TaskInfo::TaskInfo(std::string task_name) : impl_{std::make_unique<Impl>(std::move(task_name))} {}

TaskInfo::~TaskInfo() = default;

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
                           const std::map<LegateVariantCode, VariantOptions>& all_options)
{
  const auto finder = all_options.find(vid);

  add_variant(vid,
              body,
              Legion::CodeDescriptor{entry},
              finder == all_options.end() ? VariantOptions{} : finder->second);
}

const VariantInfo& TaskInfo::find_variant(LegateVariantCode vid) const
{
  return impl_->find_variant(vid);
}

bool TaskInfo::has_variant(LegateVariantCode vid) const { return impl_->has_variant(vid); }

void TaskInfo::register_task(Legion::TaskID task_id) { return impl_->register_task(task_id); }

std::ostream& operator<<(std::ostream& os, const VariantInfo& info)
{
  std::stringstream ss;
  ss << std::showbase << std::hex << reinterpret_cast<uintptr_t>(info.body) << "," << info.options;
  os << std::move(ss).str();
  return os;
}

std::ostream& operator<<(std::ostream& os, const TaskInfo& info)
{
  std::stringstream ss;
  ss << info.name() << " {";
  for (auto [vid, vinfo] : info.impl_->variants()) {
    ss << VARIANT_NAMES[vid] << ":[" << vinfo << "],";
  }
  ss << "}";
  os << std::move(ss).str();
  return os;
}

}  // namespace legate
