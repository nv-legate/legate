/* Copyright 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "core/runtime/detail/fill_launcher.h"
#include "core/data/detail/logical_store.h"
#include "core/mapping/machine.h"
#include "core/runtime/context.h"
#include "core/runtime/detail/projection.h"
#include "core/runtime/detail/runtime.h"
#include "core/utilities/buffer_builder.h"

namespace legate::detail {

FillLauncher::FillLauncher(LibraryContext* library,
                           const mapping::MachineDesc& machine,
                           int64_t tag)
  : library_(library), tag_(tag)
{
  mapper_arg_ = new BufferBuilder();
  machine.pack(*mapper_arg_);
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

void FillLauncher::pack_args() { mapper_arg_->pack<uint32_t>(0); }

std::unique_ptr<Legion::IndexFillLauncher> FillLauncher::build_index_fill(
  const Legion::Domain& launch_domain,
  LogicalStore* lhs,
  const ProjectionInfo& lhs_proj,
  LogicalStore* value)
{
  pack_args();
  auto* runtime         = Runtime::get_runtime();
  auto& provenance      = runtime->provenance_manager()->get_provenance();
  auto lhs_region_field = lhs->get_region_field();
  auto lhs_region       = lhs_region_field->region();
  auto field_id         = lhs_region_field->field_id();
  auto future_value     = value->get_future();
  auto lhs_parent       = Runtime::get_runtime()->find_parent_region(lhs_region);
  auto index_fill       = std::make_unique<Legion::IndexFillLauncher>(launch_domain,
                                                                lhs_proj.partition,
                                                                lhs_parent,
                                                                future_value,
                                                                lhs_proj.proj_id,
                                                                Legion::Predicate::TRUE_PRED,
                                                                library_->get_mapper_id(),
                                                                lhs_proj.tag,
                                                                mapper_arg_->to_legion_buffer(),
                                                                provenance.c_str());

  index_fill->add_field(field_id);
  return std::move(index_fill);
}

std::unique_ptr<Legion::FillLauncher> FillLauncher::build_single_fill(
  LogicalStore* lhs, const ProjectionInfo& lhs_proj, LogicalStore* value)
{
  pack_args();
  auto* runtime         = Runtime::get_runtime();
  auto& provenance      = runtime->provenance_manager()->get_provenance();
  auto lhs_region_field = lhs->get_region_field();
  auto lhs_region       = lhs_region_field->region();
  auto field_id         = lhs_region_field->field_id();
  auto future_value     = value->get_future();
  auto lhs_parent       = Runtime::get_runtime()->find_parent_region(lhs_region);
  auto single_fill      = std::make_unique<Legion::FillLauncher>(lhs_region,
                                                            lhs_parent,
                                                            future_value,
                                                            Legion::Predicate::TRUE_PRED,
                                                            library_->get_mapper_id(),
                                                            lhs_proj.tag,
                                                            mapper_arg_->to_legion_buffer(),
                                                            provenance.c_str());

  single_fill->add_field(field_id);
  return std::move(single_fill);
}

}  // namespace legate::detail
