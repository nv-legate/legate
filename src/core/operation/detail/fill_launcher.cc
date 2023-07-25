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
#include "core/operation/detail/projection.h"
#include "core/runtime/detail/library.h"
#include "core/runtime/detail/runtime.h"
#include "core/utilities/detail/buffer_builder.h"

namespace legate::detail {

FillLauncher::FillLauncher(const mapping::detail::Machine& machine, int64_t tag)
  : machine_(machine), tag_(tag), mapper_arg_(new BufferBuilder())
{
}

FillLauncher::~FillLauncher() { delete mapper_arg_; }

void FillLauncher::launch(const Legion::Domain& launch_domain,
                          LogicalStore* lhs,
                          const ProjectionInfo& lhs_proj,
                          LogicalStore* value)
{
  auto legion_fill_launcher = build_index_fill(launch_domain, lhs, lhs_proj, value);
  return Runtime::get_runtime()->dispatch(legion_fill_launcher.get());
}

void FillLauncher::launch_single(LogicalStore* lhs,
                                 const ProjectionInfo& lhs_proj,
                                 LogicalStore* value)
{
  auto legion_fill_launcher = build_single_fill(lhs, lhs_proj, value);
  return Runtime::get_runtime()->dispatch(legion_fill_launcher.get());
}

void FillLauncher::pack_mapper_arg(Legion::ProjectionID proj_id)
{
  machine_.pack(*mapper_arg_);
  mapper_arg_->pack<uint32_t>(Runtime::get_runtime()->get_sharding(machine_, proj_id));
}

std::unique_ptr<Legion::IndexFillLauncher> FillLauncher::build_index_fill(
  const Legion::Domain& launch_domain,
  LogicalStore* lhs,
  const ProjectionInfo& lhs_proj,
  LogicalStore* value)
{
  pack_mapper_arg(lhs_proj.proj_id);
  auto* runtime         = Runtime::get_runtime();
  auto& provenance      = runtime->provenance_manager()->get_provenance();
  auto lhs_region_field = lhs->get_region_field();
  auto lhs_region       = lhs_region_field->region();
  auto field_id         = lhs_region_field->field_id();
  auto future_value     = value->get_future();
  auto lhs_parent       = runtime->find_parent_region(lhs_region);
  auto index_fill =
    std::make_unique<Legion::IndexFillLauncher>(launch_domain,
                                                lhs_proj.partition,
                                                lhs_parent,
                                                future_value,
                                                lhs_proj.proj_id,
                                                Legion::Predicate::TRUE_PRED,
                                                runtime->core_library()->get_mapper_id(),
                                                lhs_proj.tag,
                                                mapper_arg_->to_legion_buffer(),
                                                provenance.c_str());

  index_fill->add_field(field_id);
  return std::move(index_fill);
}

std::unique_ptr<Legion::FillLauncher> FillLauncher::build_single_fill(
  LogicalStore* lhs, const ProjectionInfo& lhs_proj, LogicalStore* value)
{
  pack_mapper_arg(lhs_proj.proj_id);
  auto* runtime         = Runtime::get_runtime();
  auto& provenance      = runtime->provenance_manager()->get_provenance();
  auto lhs_region_field = lhs->get_region_field();
  auto lhs_region       = lhs_region_field->region();
  auto field_id         = lhs_region_field->field_id();
  auto future_value     = value->get_future();
  auto lhs_parent       = runtime->find_parent_region(lhs_region);
  auto single_fill =
    std::make_unique<Legion::FillLauncher>(lhs_region,
                                           lhs_parent,
                                           future_value,
                                           Legion::Predicate::TRUE_PRED,
                                           runtime->core_library()->get_mapper_id(),
                                           lhs_proj.tag,
                                           mapper_arg_->to_legion_buffer(),
                                           provenance.c_str());

  single_fill->add_field(field_id);
  return std::move(single_fill);
}

}  // namespace legate::detail
