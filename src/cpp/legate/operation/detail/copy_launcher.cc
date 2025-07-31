/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/copy_launcher.h>

#include <legate/data/detail/logical_store.h>
#include <legate/operation/detail/store_projection.h>
#include <legate/runtime/detail/library.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/runtime/detail/streaming.h>
#include <legate/utilities/detail/buffer_builder.h>
#include <legate/utilities/detail/zip.h>

namespace legate::detail {

CopyArg::CopyArg(std::uint32_t req_idx,
                 LogicalStore* store,
                 Legion::FieldID field_id,
                 Legion::PrivilegeMode privilege,
                 StoreProjection store_proj)
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

  buffer.pack<GlobalRedopID>(store_proj_.redop);
  buffer.pack<std::int32_t>(region_.get_dim());
  buffer.pack<std::uint32_t>(req_idx_);
  buffer.pack<std::uint32_t>(field_id_);
}

void CopyLauncher::add_store(SmallVector<CopyArg>& args,
                             const InternalSharedPtr<LogicalStore>& store,
                             StoreProjection store_proj,
                             Legion::PrivilegeMode privilege)
{
  const auto req_idx  = static_cast<std::uint32_t>(args.size());
  const auto field_id = store->get_region_field()->field_id();

  if (store_proj.is_key) {
    key_proj_id_ = store_proj.proj_id;
  }
  args.emplace_back(req_idx, store.get(), field_id, privilege, std::move(store_proj));
}

void CopyLauncher::add_input(const InternalSharedPtr<LogicalStore>& store,
                             StoreProjection store_proj)
{
  add_store(inputs_, store, std::move(store_proj), LEGION_READ_ONLY);
}

void CopyLauncher::add_output(const InternalSharedPtr<LogicalStore>& store,
                              StoreProjection store_proj)
{
  add_store(outputs_, store, std::move(store_proj), LEGION_WRITE_ONLY);
}

void CopyLauncher::add_inout(const InternalSharedPtr<LogicalStore>& store,
                             StoreProjection store_proj)
{
  add_store(outputs_, store, std::move(store_proj), LEGION_READ_WRITE);
}

void CopyLauncher::add_reduction(const InternalSharedPtr<LogicalStore>& store,
                                 StoreProjection store_proj)
{
  add_store(outputs_, store, std::move(store_proj), LEGION_REDUCE);
}

void CopyLauncher::add_source_indirect(const InternalSharedPtr<LogicalStore>& store,
                                       StoreProjection store_proj)
{
  add_store(source_indirect_, store, std::move(store_proj), LEGION_READ_ONLY);
}

void CopyLauncher::add_target_indirect(const InternalSharedPtr<LogicalStore>& store,
                                       StoreProjection store_proj)
{
  add_store(target_indirect_, store, std::move(store_proj), LEGION_READ_ONLY);
}

void CopyLauncher::execute(const Legion::Domain& launch_domain)
{
  BufferBuilder mapper_arg;

  pack_args(mapper_arg);

  auto&& runtime  = Runtime::get_runtime();
  auto index_copy = Legion::IndexCopyLauncher{launch_domain,
                                              Legion::Predicate::TRUE_PRED,
                                              runtime.mapper_id(),
                                              static_cast<Legion::MappingTagID>(tag_),
                                              mapper_arg.to_legion_buffer()};

  populate_copy_(index_copy);
  runtime.dispatch(index_copy);
}

void CopyLauncher::execute_single()
{
  BufferBuilder mapper_arg;

  pack_args(mapper_arg);

  auto&& runtime   = Runtime::get_runtime();
  auto single_copy = Legion::CopyLauncher{Legion::Predicate::TRUE_PRED,
                                          runtime.mapper_id(),
                                          static_cast<Legion::MappingTagID>(tag_),
                                          mapper_arg.to_legion_buffer()};

  populate_copy_(single_copy);
  runtime.dispatch(single_copy);
}

void CopyLauncher::pack_sharding_functor_id(BufferBuilder& buffer)
{
  buffer.pack<std::uint32_t>(Runtime::get_runtime().get_sharding(machine_, key_proj_id_));
}

void CopyLauncher::pack_args(BufferBuilder& buffer)
{
  buffer.pack<StreamingGeneration>(std::nullopt);
  machine_.pack(buffer);
  pack_sharding_functor_id(buffer);
  buffer.pack(priority_);

  auto pack_args = [&buffer](Span<const CopyArg> args) {
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
  constexpr auto populate_requirements = [](SmallVector<CopyArg>& args, auto& requirements) {
    requirements.reserve(args.size());
    LEGATE_CHECK(requirements.empty());
    for (auto&& arg : args) {
      requirements.emplace_back(arg.template create_requirement<is_single_v<Launcher>>());
    }
  };

  launcher.provenance = Runtime::get_runtime().get_provenance().as_string_view();

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
