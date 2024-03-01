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

#include "core/operation/detail/fill_launcher.h"

#include "core/data/detail/logical_store.h"
#include "core/mapping/machine.h"
#include "core/operation/detail/store_projection.h"
#include "core/runtime/detail/library.h"
#include "core/runtime/detail/runtime.h"
#include "core/utilities/detail/buffer_builder.h"

namespace legate::detail {

void FillLauncher::launch(const Legion::Domain& launch_domain,
                          LogicalStore* lhs,
                          const StoreProjection& lhs_proj,
                          LogicalStore* value)
{
  BufferBuilder mapper_arg;

  pack_mapper_arg(mapper_arg, lhs_proj.proj_id);

  const auto runtime      = Runtime::get_runtime();
  auto&& provenance       = runtime->provenance_manager()->get_provenance();
  auto&& lhs_region_field = lhs->get_region_field();
  auto&& lhs_region       = lhs_region_field->region();
  auto field_id           = lhs_region_field->field_id();
  auto future_value       = value->get_future();
  auto lhs_parent         = runtime->find_parent_region(lhs_region);
  auto index_fill         = Legion::IndexFillLauncher{
    launch_domain,
    lhs_proj.partition,
    std::move(lhs_parent),
    std::move(future_value),
    lhs_proj.proj_id,
    Legion::Predicate::TRUE_PRED,
    runtime->core_library()->get_mapper_id(),
    static_cast<Legion::MappingTagID>(lhs_proj.is_key ? LEGATE_CORE_KEY_STORE_TAG : 0),
    mapper_arg.to_legion_buffer(),
    provenance.c_str()};

  index_fill.add_field(field_id);
  runtime->dispatch(index_fill);
}

void FillLauncher::launch(const Legion::Domain& launch_domain,
                          LogicalStore* lhs,
                          const StoreProjection& lhs_proj,
                          const Scalar& value)
{
  BufferBuilder mapper_arg;

  pack_mapper_arg(mapper_arg, lhs_proj.proj_id);

  const auto runtime      = Runtime::get_runtime();
  auto&& provenance       = runtime->provenance_manager()->get_provenance();
  auto&& lhs_region_field = lhs->get_region_field();
  auto&& lhs_region       = lhs_region_field->region();
  auto field_id           = lhs_region_field->field_id();
  auto lhs_parent         = runtime->find_parent_region(lhs_region);
  auto index_fill         = Legion::IndexFillLauncher{
    launch_domain,
    lhs_proj.partition,
    std::move(lhs_parent),
    Legion::UntypedBuffer{value.data(), value.size()},
    lhs_proj.proj_id,
    Legion::Predicate::TRUE_PRED,
    runtime->core_library()->get_mapper_id(),
    static_cast<Legion::MappingTagID>(lhs_proj.is_key ? LEGATE_CORE_KEY_STORE_TAG : 0),
    mapper_arg.to_legion_buffer(),
    provenance.c_str()};

  index_fill.add_field(field_id);
  runtime->dispatch(index_fill);
}

void FillLauncher::launch_single(LogicalStore* lhs,
                                 const StoreProjection& lhs_proj,
                                 LogicalStore* value)
{
  BufferBuilder mapper_arg;

  pack_mapper_arg(mapper_arg, lhs_proj.proj_id);

  const auto runtime      = Runtime::get_runtime();
  auto&& provenance       = runtime->provenance_manager()->get_provenance();
  auto&& lhs_region_field = lhs->get_region_field();
  auto&& lhs_region       = lhs_region_field->region();
  auto field_id           = lhs_region_field->field_id();
  auto future_value       = value->get_future();
  auto lhs_parent         = runtime->find_parent_region(lhs_region);
  auto single_fill        = Legion::FillLauncher{
    lhs_region,
    std::move(lhs_parent),
    std::move(future_value),
    Legion::Predicate::TRUE_PRED,
    runtime->core_library()->get_mapper_id(),
    static_cast<Legion::MappingTagID>(lhs_proj.is_key ? LEGATE_CORE_KEY_STORE_TAG : 0),
    mapper_arg.to_legion_buffer(),
    provenance.c_str()};

  single_fill.add_field(field_id);
  runtime->dispatch(single_fill);
}

void FillLauncher::launch_single(LogicalStore* lhs,
                                 const StoreProjection& lhs_proj,
                                 const Scalar& value)
{
  BufferBuilder mapper_arg;

  pack_mapper_arg(mapper_arg, lhs_proj.proj_id);

  const auto runtime      = Runtime::get_runtime();
  auto&& provenance       = runtime->provenance_manager()->get_provenance();
  auto&& lhs_region_field = lhs->get_region_field();
  auto&& lhs_region       = lhs_region_field->region();
  auto field_id           = lhs_region_field->field_id();
  auto lhs_parent         = runtime->find_parent_region(lhs_region);
  auto single_fill        = Legion::FillLauncher{
    lhs_region,
    std::move(lhs_parent),
    Legion::UntypedBuffer{value.data(), value.size()},
    Legion::Predicate::TRUE_PRED,
    runtime->core_library()->get_mapper_id(),
    static_cast<Legion::MappingTagID>(lhs_proj.is_key ? LEGATE_CORE_KEY_STORE_TAG : 0),
    mapper_arg.to_legion_buffer(),
    provenance.c_str()};

  single_fill.add_field(field_id);
  runtime->dispatch(single_fill);
}

void FillLauncher::pack_mapper_arg(BufferBuilder& buffer, Legion::ProjectionID proj_id)
{
  machine_.pack(buffer);
  buffer.pack<std::uint32_t>(Runtime::get_runtime()->get_sharding(machine_, proj_id));
}

}  // namespace legate::detail
