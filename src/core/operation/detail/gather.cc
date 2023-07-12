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

#include "core/operation/detail/gather.h"

#include "core/operation/detail/copy_launcher.h"
#include "core/partitioning/constraint.h"
#include "core/partitioning/constraint_solver.h"
#include "core/partitioning/partition.h"
#include "core/partitioning/partitioner.h"

namespace legate::detail {

Gather::Gather(std::shared_ptr<LogicalStore> target,
               std::shared_ptr<LogicalStore> source,
               std::shared_ptr<LogicalStore> source_indirect,
               int64_t unique_id,
               mapping::MachineDesc&& machine)
  : Operation(unique_id, std::move(machine)),
    target_{target.get(), declare_partition()},
    source_{source.get(), declare_partition()},
    source_indirect_{source_indirect.get(), declare_partition()},
    constraint_(legate::align(target_.variable, source_indirect_.variable))
{
  record_partition(target_.variable, std::move(target));
  record_partition(source_.variable, std::move(source));
  record_partition(source_indirect_.variable, std::move(source_indirect));
}

void Gather::validate()
{
  if (source_.store->type() != target_.store->type()) {
    throw std::invalid_argument("Source and targets must have the same type");
  }
  auto validate_store = [](auto* store) {
    if (store->unbound() || store->has_scalar_storage() || store->transformed()) {
      throw std::invalid_argument(
        "Gather accepts only normal, untransformed, region-backed stores");
    }
  };
  validate_store(target_.store);
  validate_store(source_.store);
  validate_store(source_indirect_.store);

  if (!is_point_type(source_indirect_.store->type(), source_.store->dim())) {
    throw std::invalid_argument("Indirection store should contain " +
                                std::to_string(source_.store->dim()) + "-D points");
  }

  constraint_->validate();
}

void Gather::launch(Strategy* p_strategy)
{
  auto& strategy = *p_strategy;
  CopyLauncher launcher(machine_);
  auto launch_domain = strategy.launch_domain(this);

  launcher.add_input(source_.store, create_projection_info(strategy, launch_domain, source_));
  launcher.add_source_indirect(source_indirect_.store,
                               create_projection_info(strategy, launch_domain, source_indirect_));
  launcher.add_output(target_.store, create_projection_info(strategy, launch_domain, target_));
  launcher.set_source_indirect_out_of_range(out_of_range_);

  if (launch_domain != nullptr) {
    return launcher.execute(*launch_domain);
  } else {
    return launcher.execute_single();
  }
}

void Gather::add_to_solver(ConstraintSolver& solver)
{
  solver.add_constraint(constraint_.get());
  solver.add_partition_symbol(target_.variable);
  solver.add_partition_symbol(source_.variable);
  solver.add_partition_symbol(source_indirect_.variable);
}

std::string Gather::to_string() const { return "Gather:" + std::to_string(unique_id_); }

}  // namespace legate::detail
