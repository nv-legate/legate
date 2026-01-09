/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/fill_launcher.h>

#include <legate/data/detail/logical_store.h>
#include <legate/mapping/machine.h>
#include <legate/operation/detail/store_projection.h>
#include <legate/runtime/detail/library.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/runtime/detail/streaming/generation.h>
#include <legate/utilities/detail/buffer_builder.h>
#include <legate/utilities/detail/core_ids.h>

namespace legate::detail {

namespace {

std::tuple<Legion::LogicalRegion, Legion::LogicalRegion, Legion::FieldID> prepare_lhs(
  const LogicalStore* lhs)
{
  const auto& lhs_region_field = lhs->get_region_field();
  const auto& lhs_region       = lhs_region_field->region();
  auto field_id                = lhs_region_field->field_id();
  auto lhs_parent              = Runtime::get_runtime().find_parent_region(lhs_region);
  return {lhs_region, lhs_parent, field_id};
}

}  // namespace

void FillLauncher::launch(const Legion::Domain& launch_domain,
                          LogicalStore* lhs,
                          const StoreProjection& lhs_proj,
                          Legion::Future value)
{
  auto mapper_arg = pack_mapper_arg_(lhs_proj.proj_id);

  auto&& runtime                 = Runtime::get_runtime();
  auto [_, lhs_parent, field_id] = prepare_lhs(lhs);
  auto index_fill                = Legion::IndexFillLauncher{
    launch_domain,
    lhs_proj.partition,
    std::move(lhs_parent),
    std::move(value),
    lhs_proj.proj_id,
    Legion::Predicate::TRUE_PRED,
    runtime.mapper_id(),
    lhs_proj.is_key ? static_cast<Legion::MappingTagID>(CoreMappingTag::KEY_STORE) : 0,
    mapper_arg.to_legion_buffer()};

  index_fill.provenance = runtime.get_provenance().as_string_view();
  index_fill.add_field(field_id);
  runtime.dispatch(index_fill);
}

void FillLauncher::launch_single(LogicalStore* lhs,
                                 const StoreProjection& lhs_proj,
                                 Legion::Future value)
{
  auto mapper_arg = pack_mapper_arg_(lhs_proj.proj_id);

  auto&& runtime                          = Runtime::get_runtime();
  auto [lhs_region, lhs_parent, field_id] = prepare_lhs(lhs);
  auto single_fill                        = Legion::FillLauncher{
    lhs_region,
    std::move(lhs_parent),
    std::move(value),
    Legion::Predicate::TRUE_PRED,
    runtime.mapper_id(),
    lhs_proj.is_key ? static_cast<Legion::MappingTagID>(CoreMappingTag::KEY_STORE) : 0,
    mapper_arg.to_legion_buffer()};

  single_fill.provenance = runtime.get_provenance().as_string_view();
  single_fill.add_field(field_id);
  runtime.dispatch(single_fill);
}

BufferBuilder FillLauncher::pack_mapper_arg_(Legion::ProjectionID proj_id) const
{
  BufferBuilder buffer;

  buffer.pack<StreamingGeneration>(std::nullopt);
  machine_.pack(buffer);
  buffer.pack<std::uint32_t>(Runtime::get_runtime().get_sharding(machine_, proj_id));
  buffer.pack(priority_);
  return buffer;
}

}  // namespace legate::detail
