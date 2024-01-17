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

#include "core/operation/detail/reduce.h"

#include "core/data/detail/logical_store.h"
#include "core/operation/detail/store_projection.h"
#include "core/operation/detail/task_launcher.h"
#include "core/operation/projection.h"
#include "core/partitioning/detail/constraint.h"
#include "core/partitioning/detail/constraint_solver.h"
#include "core/partitioning/detail/partitioner.h"
#include "core/partitioning/partition.h"
#include "core/runtime/detail/library.h"
#include "core/runtime/detail/runtime.h"

namespace legate::detail {

Reduce::Reduce(const Library* library,
               InternalSharedPtr<LogicalStore> store,
               InternalSharedPtr<LogicalStore> out_store,
               int64_t task_id,
               uint64_t unique_id,
               int32_t radix,
               mapping::detail::Machine&& machine)
  : Operation{unique_id, std::move(machine)},
    radix_{radix},
    library_{library},
    task_id_{task_id},
    input_{std::move(store)},
    output_{std::move(out_store)},
    input_part_{find_or_declare_partition(input_)},
    output_part_{declare_partition()}
{
  record_partition(input_part_, input_);
  record_partition(output_part_, output_);
}

void Reduce::launch(Strategy* p_strategy)
{
  auto& strategy     = *p_strategy;
  auto launch_domain = strategy.launch_domain(this);
  auto n_tasks       = launch_domain.is_valid() ? launch_domain.get_volume() : 1;

  if (LegateDefined(LEGATE_USE_DEBUG)) {
    assert(!launch_domain.is_valid() || launch_domain.dim == 1);
  }

  auto input_part      = strategy[input_part_];
  auto input_partition = create_store_partition(input_, input_part);

  // generating projection functions to use in tree_reduction task
  std::vector<proj::SymbolicPoint> projections;
  if (n_tasks > 1) {
    projections.reserve(static_cast<size_t>(radix_));
    for (int32_t i = 0; i < radix_; i++) {
      projections.emplace_back(std::initializer_list<SymbolicExpr>{SymbolicExpr{0, radix_, i}});
    }
  }

  auto to_array_arg = [](auto&& arg) {
    return std::make_unique<BaseArrayArg>(std::forward<decltype(arg)>(arg));
  };

  const auto runtime = detail::Runtime::get_runtime();

  InternalSharedPtr<LogicalStore> new_output;
  bool done = false;

  do {
    auto launcher =
      detail::TaskLauncher{library_, machine_, provenance_, task_id_, LEGATE_CORE_TREE_REDUCE_TAG};

    if (n_tasks > 1) {
      // if there are more than 1 sub-task, we add several slices of the input
      // for each sub-task
      for (auto& projection : projections) {
        auto store_proj = input_partition->create_store_projection(launch_domain, projection);

        launcher.add_input(to_array_arg(
          std::make_unique<RegionFieldArg>(input_.get(), LEGION_READ_ONLY, std::move(store_proj))));
      }
    } else {
      // otherwise we just add an entire input region to the task
      auto store_proj = input_partition->create_store_projection(launch_domain);

      launcher.add_input(to_array_arg(
        std::make_unique<RegionFieldArg>(input_.get(), LEGION_READ_ONLY, std::move(store_proj))));
    }

    // calculating #of sub-tasks in the reduction task
    n_tasks = (n_tasks + radix_ - 1) / radix_;
    done    = n_tasks == 1;

    // adding output region
    auto field_space = runtime->create_field_space();
    // if this is not the last iteration of the while loop, we generate
    // a new output region
    if (n_tasks != 1) {
      new_output = runtime->create_store(input_->type(), 1);
      launcher.add_output(
        to_array_arg(std::make_unique<OutputRegionArg>(new_output.get(), field_space)));
    } else {
      launcher.add_output(
        to_array_arg(std::make_unique<OutputRegionArg>(output_.get(), field_space)));
    }

    launch_domain = Domain{DomainPoint{0}, DomainPoint{static_cast<coord_t>(n_tasks - 1)}};
    auto result   = launcher.execute(launch_domain);

    if (n_tasks != 1) {
      auto weighted = Weighted{result, launch_domain};

      new_output->set_key_partition(machine_, &weighted);
      auto output_partition =
        create_store_partition(new_output, make_internal_shared<Weighted>(std::move(weighted)));
      input_          = new_output;
      input_partition = output_partition;
    }
  } while (!done);
}

void Reduce::add_to_solver(detail::ConstraintSolver& solver)
{
  solver.add_partition_symbol(output_part_, AccessMode::WRITE);
  solver.add_partition_symbol(input_part_, AccessMode::READ);
}

std::string Reduce::to_string() const { return "Reduce:" + std::to_string(unique_id_); }

}  // namespace legate::detail
