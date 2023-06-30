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

#include "core/operation/detail/fill.h"

#include "core/data/detail/logical_store.h"
#include "core/operation/detail/fill_launcher.h"
#include "core/operation/detail/projection.h"
#include "core/partitioning/constraint_solver.h"
#include "core/partitioning/partitioner.h"

namespace legate::detail {

Fill::Fill(std::shared_ptr<LogicalStore>&& lhs,
           std::shared_ptr<LogicalStore>&& value,
           int64_t unique_id,
           mapping::MachineDesc&& machine)
  : Operation(unique_id, std::move(machine)),
    lhs_var_(declare_partition()),
    lhs_(std::move(lhs)),
    value_(std::move(value))
{
  store_mappings_[*lhs_var_] = lhs_.get();
  if (lhs_->unbound() || lhs_->has_scalar_storage())
    throw std::runtime_error("Fill lhs must be a normal, region-backed store");

  if (!value_->has_scalar_storage())
    throw std::runtime_error("Fill value should be a Future-back store");
}

void Fill::validate() {}

void Fill::launch(Strategy* strategy)
{
  FillLauncher launcher(machine_);
  auto launch_domain = strategy->launch_domain(this);
  auto part          = (*strategy)[lhs_var_];
  auto lhs_proj      = lhs_->create_partition(part)->create_projection_info(launch_domain);
  lhs_->set_key_partition(machine(), part.get());

  if (nullptr == launch_domain)
    launcher.launch_single(lhs_.get(), *lhs_proj, value_.get());
  else
    launcher.launch(*launch_domain, lhs_.get(), *lhs_proj, value_.get());
}

std::string Fill::to_string() const { return "Fill:" + std::to_string(unique_id_); }

void Fill::add_to_solver(ConstraintSolver& solver) { solver.add_partition_symbol(lhs_var_); }

}  // namespace legate::detail
