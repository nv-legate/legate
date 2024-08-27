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

#include "core/operation/detail/copy_launcher.h"

#include "core/data/detail/logical_store.h"
#include "core/operation/detail/store_projection.h"
#include "core/runtime/detail/library.h"
#include "core/runtime/detail/runtime.h"
#include "core/utilities/detail/buffer_builder.h"
#include "core/utilities/detail/zip.h"

namespace legate::detail {

CopyArg::CopyArg(std::uint32_t req_idx,
                 LogicalStore* store,
                 Legion::FieldID field_id,
                 Legion::PrivilegeMode privilege,
                 std::unique_ptr<StoreProjection> store_proj)
  : req_idx_{req_idx},
    store_{store},
    region_{store_->get_region_field()->region()},
    field_id_{field_id},
    privilege_{privilege},
    store_proj_{std::move(store_proj)}
{
}

void CopyArg::pack(BufferBuilder& buffer) const
{
  store_->pack(buffer);

  buffer.pack<GlobalRedopID>(store_proj_->redop);
  buffer.pack<std::int32_t>(region_.get_dim());
  buffer.pack<std::uint32_t>(req_idx_);
  buffer.pack<std::uint32_t>(field_id_);
}

void CopyLauncher::add_store(std::vector<CopyArg>& args,
                             const InternalSharedPtr<LogicalStore>& store,
                             std::unique_ptr<StoreProjection> store_proj,
                             Legion::PrivilegeMode privilege)
{
  const auto req_idx  = static_cast<std::uint32_t>(args.size());
  const auto field_id = store->get_region_field()->field_id();

  if (store_proj->is_key) {
    key_proj_id_ = store_proj->proj_id;
  }
  args.emplace_back(req_idx, store.get(), field_id, privilege, std::move(store_proj));
}

void CopyLauncher::add_input(const InternalSharedPtr<LogicalStore>& store,
                             std::unique_ptr<StoreProjection> store_proj)
{
  add_store(inputs_, store, std::move(store_proj), LEGION_READ_ONLY);
}

void CopyLauncher::add_output(const InternalSharedPtr<LogicalStore>& store,
                              std::unique_ptr<StoreProjection> store_proj)
{
  add_store(outputs_, store, std::move(store_proj), LEGION_WRITE_ONLY);
}

void CopyLauncher::add_inout(const InternalSharedPtr<LogicalStore>& store,
                             std::unique_ptr<StoreProjection> store_proj)
{
  add_store(outputs_, store, std::move(store_proj), LEGION_READ_WRITE);
}

void CopyLauncher::add_reduction(const InternalSharedPtr<LogicalStore>& store,
                                 std::unique_ptr<StoreProjection> store_proj)
{
  add_store(outputs_, store, std::move(store_proj), LEGION_REDUCE);
}

void CopyLauncher::add_source_indirect(const InternalSharedPtr<LogicalStore>& store,
                                       std::unique_ptr<StoreProjection> store_proj)
{
  add_store(source_indirect_, store, std::move(store_proj), LEGION_READ_ONLY);
}

void CopyLauncher::add_target_indirect(const InternalSharedPtr<LogicalStore>& store,
                                       std::unique_ptr<StoreProjection> store_proj)
{
  add_store(target_indirect_, store, std::move(store_proj), LEGION_READ_ONLY);
}

void CopyLauncher::execute(const Legion::Domain& launch_domain)
{
  BufferBuilder mapper_arg;

  pack_args(mapper_arg);

  const auto runtime = Runtime::get_runtime();
  auto index_copy    = Legion::IndexCopyLauncher{launch_domain,
                                              Legion::Predicate::TRUE_PRED,
                                              runtime->mapper_id(),
                                              static_cast<Legion::MappingTagID>(tag_),
                                              mapper_arg.to_legion_buffer()};

  populate_copy_(index_copy);
  runtime->dispatch(index_copy);
}

void CopyLauncher::execute_single()
{
  BufferBuilder mapper_arg;

  pack_args(mapper_arg);

  const auto runtime = Runtime::get_runtime();
  auto single_copy   = Legion::CopyLauncher{Legion::Predicate::TRUE_PRED,
                                          runtime->mapper_id(),
                                          static_cast<Legion::MappingTagID>(tag_),
                                          mapper_arg.to_legion_buffer()};

  populate_copy_(single_copy);
  runtime->dispatch(single_copy);
}

void CopyLauncher::pack_sharding_functor_id(BufferBuilder& buffer)
{
  buffer.pack<std::uint32_t>(Runtime::get_runtime()->get_sharding(machine_, key_proj_id_));
}

void CopyLauncher::pack_args(BufferBuilder& buffer)
{
  machine_.pack(buffer);
  pack_sharding_functor_id(buffer);
  buffer.pack(priority_);

  auto pack_args = [&buffer](const std::vector<CopyArg>& args) {
    buffer.pack<std::uint32_t>(static_cast<std::uint32_t>(args.size()));
    for (auto&& arg : args) {
      arg.pack(buffer);
    }
  };
  pack_args(inputs_);
  pack_args(outputs_);
  pack_args(source_indirect_);
  pack_args(target_indirect_);
}

namespace {

template <typename Launcher>
constexpr bool is_single_v = false;
template <>
constexpr bool is_single_v<Legion::CopyLauncher> = true;

}  // namespace

template <typename Launcher>
void CopyLauncher::populate_copy_(Launcher& launcher)
{
  constexpr auto populate_requirements = [](std::vector<CopyArg>& args, auto& requirements) {
    requirements.resize(args.size());
    for (auto&& [arg, req] : detail::zip_equal(args, requirements)) {
      arg.template populate_requirement<is_single_v<Launcher>>(req);
    }
  };

  launcher.provenance = Runtime::get_runtime()->get_provenance().as_string_view();

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
