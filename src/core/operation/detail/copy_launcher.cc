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

#include "core/operation/detail/copy_launcher.h"

#include "core/data/detail/logical_store.h"
#include "core/operation/detail/launcher_arg.h"
#include "core/operation/detail/projection.h"
#include "core/runtime/detail/library.h"
#include "core/runtime/detail/runtime.h"
#include "core/utilities/detail/buffer_builder.h"

namespace legate::detail {

struct CopyArg : public ArgWrapper {
 public:
  CopyArg(uint32_t req_idx,
          LogicalStore* store,
          Legion::FieldID field_id,
          Legion::PrivilegeMode privilege,
          std::unique_ptr<ProjectionInfo> proj_info);

 public:
  void pack(BufferBuilder& buffer) const override;

 public:
  template <bool SINGLE>
  void populate_requirement(Legion::RegionRequirement& requirement)
  {
    proj_info_->template populate_requirement<SINGLE>(
      requirement, region_, {field_id_}, privilege_);
  }

 public:
  ~CopyArg() {}

 private:
  uint32_t req_idx_;
  LogicalStore* store_;
  Legion::LogicalRegion region_;
  Legion::FieldID field_id_;
  Legion::PrivilegeMode privilege_;
  std::unique_ptr<ProjectionInfo> proj_info_;
};

CopyArg::CopyArg(uint32_t req_idx,
                 LogicalStore* store,
                 Legion::FieldID field_id,
                 Legion::PrivilegeMode privilege,
                 std::unique_ptr<ProjectionInfo> proj_info)
  : req_idx_(req_idx),
    store_(store),
    region_(store_->get_region_field()->region()),
    field_id_(field_id),
    privilege_(privilege),
    proj_info_(std::move(proj_info))
{
}

void CopyArg::pack(BufferBuilder& buffer) const
{
  store_->pack(buffer);

  buffer.pack<int32_t>(proj_info_->redop);
  buffer.pack<int32_t>(region_.get_dim());
  buffer.pack<uint32_t>(req_idx_);
  buffer.pack<uint32_t>(field_id_);
}

CopyLauncher::CopyLauncher(const mapping::detail::Machine& machine, int64_t tag)
  : machine_(machine), tag_(tag)
{
  mapper_arg_ = new BufferBuilder();
  machine_.pack(*mapper_arg_);
}

CopyLauncher::~CopyLauncher()
{
  delete mapper_arg_;
  for (auto& arg : inputs_) delete arg;
  for (auto& arg : outputs_) delete arg;
  for (auto& arg : source_indirect_) delete arg;
  for (auto& arg : target_indirect_) delete arg;
}

void CopyLauncher::add_store(std::vector<CopyArg*>& args,
                             detail::LogicalStore* store,
                             std::unique_ptr<ProjectionInfo> proj_info,
                             Legion::PrivilegeMode privilege)
{
  uint32_t req_idx  = args.size();
  auto region_field = store->get_region_field();
  auto region       = region_field->region();
  auto field_id     = region_field->field_id();
  if (LEGATE_CORE_KEY_STORE_TAG == proj_info->tag) key_proj_id_ = proj_info->proj_id;
  args.push_back(new CopyArg(req_idx, store, field_id, privilege, std::move(proj_info)));
}

void CopyLauncher::add_input(detail::LogicalStore* store, std::unique_ptr<ProjectionInfo> proj_info)
{
  add_store(inputs_, store, std::move(proj_info), LEGION_READ_ONLY);
}

void CopyLauncher::add_output(detail::LogicalStore* store,
                              std::unique_ptr<ProjectionInfo> proj_info)
{
  add_store(outputs_, store, std::move(proj_info), LEGION_WRITE_ONLY);
}

void CopyLauncher::add_inout(detail::LogicalStore* store, std::unique_ptr<ProjectionInfo> proj_info)
{
  add_store(outputs_, store, std::move(proj_info), LEGION_READ_WRITE);
}

void CopyLauncher::add_reduction(detail::LogicalStore* store,
                                 std::unique_ptr<ProjectionInfo> proj_info)
{
  add_store(outputs_, store, std::move(proj_info), LEGION_REDUCE);
}
void CopyLauncher::add_source_indirect(detail::LogicalStore* store,
                                       std::unique_ptr<ProjectionInfo> proj_info)
{
  add_store(source_indirect_, store, std::move(proj_info), LEGION_READ_ONLY);
}

void CopyLauncher::add_target_indirect(detail::LogicalStore* store,
                                       std::unique_ptr<ProjectionInfo> proj_info)
{
  add_store(target_indirect_, store, std::move(proj_info), LEGION_READ_ONLY);
}

void CopyLauncher::execute(const Legion::Domain& launch_domain)
{
  auto legion_copy_launcher = build_index_copy(launch_domain);
  return Runtime::get_runtime()->dispatch(legion_copy_launcher.get());
}

void CopyLauncher::execute_single()
{
  auto legion_copy_launcher = build_single_copy();
  return Runtime::get_runtime()->dispatch(legion_copy_launcher.get());
}

void CopyLauncher::pack_sharding_functor_id()
{
  mapper_arg_->pack<uint32_t>(Runtime::get_runtime()->get_sharding(machine_, key_proj_id_));
}

void CopyLauncher::pack_args()
{
  pack_sharding_functor_id();

  auto pack_args = [&](const std::vector<CopyArg*>& args) {
    mapper_arg_->pack<uint32_t>(args.size());
    for (auto& arg : args) arg->pack(*mapper_arg_);
  };
  pack_args(inputs_);
  pack_args(outputs_);
  pack_args(source_indirect_);
  pack_args(target_indirect_);
}

namespace {

template <class Launcher>
constexpr bool is_single = false;
template <>
constexpr bool is_single<Legion::CopyLauncher> = true;
template <>
constexpr bool is_single<Legion::IndexCopyLauncher> = false;

}  // namespace

template <class Launcher>
void CopyLauncher::populate_copy(Launcher* launcher)
{
  auto populate_requirements = [&](auto& args, auto& requirements) {
    requirements.resize(args.size());
    for (uint32_t idx = 0; idx < args.size(); ++idx) {
      auto& req = requirements[idx];
      auto& arg = args[idx];
      arg->template populate_requirement<is_single<Launcher>>(req);
    }
  };

  populate_requirements(inputs_, launcher->src_requirements);
  populate_requirements(outputs_, launcher->dst_requirements);
  populate_requirements(source_indirect_, launcher->src_indirect_requirements);
  populate_requirements(target_indirect_, launcher->dst_indirect_requirements);

  launcher->src_indirect_is_range.resize(source_indirect_.size(), false);
  launcher->dst_indirect_is_range.resize(target_indirect_.size(), false);

  launcher->possible_src_indirect_out_of_range = source_indirect_out_of_range_;
  launcher->possible_dst_indirect_out_of_range = target_indirect_out_of_range_;
}

std::unique_ptr<Legion::IndexCopyLauncher> CopyLauncher::build_index_copy(
  const Legion::Domain& launch_domain)
{
  pack_args();
  auto* runtime    = Runtime::get_runtime();
  auto& provenance = runtime->provenance_manager()->get_provenance();
  auto index_copy =
    std::make_unique<Legion::IndexCopyLauncher>(launch_domain,
                                                Legion::Predicate::TRUE_PRED,
                                                runtime->core_library()->get_mapper_id(),
                                                tag_,
                                                mapper_arg_->to_legion_buffer(),
                                                provenance.c_str());

  populate_copy(index_copy.get());
  return std::move(index_copy);
}

std::unique_ptr<Legion::CopyLauncher> CopyLauncher::build_single_copy()
{
  pack_args();
  auto* runtime    = Runtime::get_runtime();
  auto& provenance = runtime->provenance_manager()->get_provenance();
  auto single_copy =
    std::make_unique<Legion::CopyLauncher>(Legion::Predicate::TRUE_PRED,
                                           runtime->core_library()->get_mapper_id(),
                                           tag_,
                                           mapper_arg_->to_legion_buffer(),
                                           provenance.c_str());

  populate_copy(single_copy.get());
  return std::move(single_copy);
}

}  // namespace legate::detail
