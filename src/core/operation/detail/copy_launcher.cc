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

struct CopyArg final : public Serializable {
 public:
  CopyArg(uint32_t req_idx,
          LogicalStore* store,
          Legion::FieldID field_id,
          Legion::PrivilegeMode privilege,
          std::unique_ptr<ProjectionInfo> proj_info);

  void pack(BufferBuilder& buffer) const override;

  template <bool SINGLE>
  void populate_requirement(Legion::RegionRequirement& requirement)
  {
    proj_info_->template populate_requirement<SINGLE>(
      requirement, region_, {field_id_}, privilege_);
  }

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
  : req_idx_{req_idx},
    store_{store},
    region_{store_->get_region_field()->region()},
    field_id_{field_id},
    privilege_{privilege},
    proj_info_{std::move(proj_info)}
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

CopyLauncher::~CopyLauncher()
{
  for (auto& arg : inputs_) {
    delete arg;
  }
  for (auto& arg : outputs_) {
    delete arg;
  }
  for (auto& arg : source_indirect_) {
    delete arg;
  }
  for (auto& arg : target_indirect_) {
    delete arg;
  }
}

void CopyLauncher::add_store(std::vector<CopyArg*>& args,
                             detail::LogicalStore* store,
                             std::unique_ptr<ProjectionInfo> proj_info,
                             Legion::PrivilegeMode privilege)
{
  auto req_idx      = static_cast<uint32_t>(args.size());
  auto region_field = store->get_region_field();
  auto field_id     = region_field->field_id();

  if (proj_info->is_key) {
    key_proj_id_ = proj_info->proj_id;
  }
  args.emplace_back(new CopyArg{req_idx, store, field_id, privilege, std::move(proj_info)});
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
  BufferBuilder mapper_arg;

  pack_args(mapper_arg);

  const auto runtime = Runtime::get_runtime();
  auto&& provenance  = runtime->provenance_manager()->get_provenance();
  auto index_copy    = Legion::IndexCopyLauncher{launch_domain,
                                              Legion::Predicate::TRUE_PRED,
                                              runtime->core_library()->get_mapper_id(),
                                              static_cast<Legion::MappingTagID>(tag_),
                                              mapper_arg.to_legion_buffer(),
                                              provenance.c_str()};
  populate_copy(index_copy);
  runtime->dispatch(index_copy);
}

void CopyLauncher::execute_single()
{
  BufferBuilder mapper_arg;

  pack_args(mapper_arg);

  const auto runtime = Runtime::get_runtime();
  auto&& provenance  = runtime->provenance_manager()->get_provenance();
  auto single_copy   = Legion::CopyLauncher{Legion::Predicate::TRUE_PRED,
                                          runtime->core_library()->get_mapper_id(),
                                          static_cast<Legion::MappingTagID>(tag_),
                                          mapper_arg.to_legion_buffer(),
                                          provenance.c_str()};
  populate_copy(single_copy);
  runtime->dispatch(single_copy);
}

void CopyLauncher::pack_sharding_functor_id(BufferBuilder& buffer)
{
  buffer.pack<uint32_t>(Runtime::get_runtime()->get_sharding(machine_, key_proj_id_));
}

void CopyLauncher::pack_args(BufferBuilder& buffer)
{
  machine_.pack(buffer);
  pack_sharding_functor_id(buffer);

  auto pack_args = [&buffer](const std::vector<CopyArg*>& args) {
    buffer.pack<uint32_t>(static_cast<uint32_t>(args.size()));
    for (auto& arg : args) {
      arg->pack(buffer);
    }
  };
  pack_args(inputs_);
  pack_args(outputs_);
  pack_args(source_indirect_);
  pack_args(target_indirect_);
}

namespace {

template <class Launcher>
constexpr bool is_single_v = false;
template <>
constexpr bool is_single_v<Legion::CopyLauncher> = true;
template <>
constexpr bool is_single_v<Legion::IndexCopyLauncher> = false;

}  // namespace

template <class Launcher>
void CopyLauncher::populate_copy(Launcher& launcher)
{
  auto populate_requirements = [&](auto& args, auto& requirements) {
    requirements.resize(args.size());
    for (uint32_t idx = 0; idx < args.size(); ++idx) {
      auto& req = requirements[idx];
      auto& arg = args[idx];
      arg->template populate_requirement<is_single_v<Launcher>>(req);
    }
  };

  populate_requirements(inputs_, launcher.src_requirements);
  populate_requirements(outputs_, launcher.dst_requirements);
  populate_requirements(source_indirect_, launcher.src_indirect_requirements);
  populate_requirements(target_indirect_, launcher.dst_indirect_requirements);

  launcher.src_indirect_is_range.resize(source_indirect_.size(), false);
  launcher.dst_indirect_is_range.resize(target_indirect_.size(), false);

  launcher.possible_src_indirect_out_of_range = source_indirect_out_of_range_;
  launcher.possible_dst_indirect_out_of_range = target_indirect_out_of_range_;
}

}  // namespace legate::detail
