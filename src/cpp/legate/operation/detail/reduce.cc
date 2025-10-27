/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/reduce.h>

#include <legate/data/detail/logical_store.h>
#include <legate/data/detail/logical_store_partition.h>
#include <legate/operation/detail/store_projection.h>
#include <legate/operation/detail/task_launcher.h>
#include <legate/operation/projection.h>
#include <legate/partitioning/detail/constraint.h>
#include <legate/partitioning/detail/constraint_solver.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/partitioning/detail/partitioner.h>
#include <legate/runtime/detail/library.h>
#include <legate/runtime/detail/region_manager.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/detail/core_ids.h>

#include <cstddef>

namespace legate::detail {

Reduce::Reduce(const Library& library,
               InternalSharedPtr<LogicalStore> store,
               InternalSharedPtr<LogicalStore> out_store,
               LocalTaskID task_id,
               std::uint64_t unique_id,
               std::int32_t radix,
               std::int32_t priority,
               mapping::detail::Machine machine)
  : Operation{unique_id, priority, std::move(machine)},
    radix_{radix},
    library_{library},
    task_id_{task_id},
    input_{std::move(store)},
    output_{std::move(out_store)},
    input_part_{find_or_declare_partition(input_)},
    output_part_{declare_partition()}
{
  record_partition_(input_part_, input_, AccessMode::READ);
  record_partition_(output_part_, output_, AccessMode::WRITE);
}

void Reduce::launch(Strategy* p_strategy)
{
  auto& strategy = *p_strategy;
  // Copy is deliberate, launch_domain is updated in the loop
  auto launch_domain = strategy.launch_domain(*this);
  auto n_tasks       = launch_domain.is_valid() ? launch_domain.get_volume() : 1;

  LEGATE_ASSERT(!launch_domain.is_valid() || launch_domain.dim == 1);

  auto input_part      = strategy[*input_part_];
  auto input_partition = create_store_partition(input_, input_part);

  // generating projection functions to use in tree_reduction task
  std::vector<proj::SymbolicPoint> projections;
  if (n_tasks > 1) {
    projections.reserve(static_cast<std::size_t>(radix_));
    for (std::int32_t i = 0; i < radix_; i++) {
      projections.emplace_back(std::initializer_list<SymbolicExpr>{SymbolicExpr{0, radix_, i}});
    }
  }

  auto&& runtime = detail::Runtime::get_runtime();

  InternalSharedPtr<LogicalStore> new_output;
  bool done = false;

  do {
    auto launcher =
      detail::TaskLauncher{library_,
                           machine_,
                           parallel_policy(),
                           provenance_,
                           task_id_,
                           static_cast<Legion::MappingTagID>(CoreMappingTag::TREE_REDUCE)};

    launcher.set_priority(priority());

    if (n_tasks > 1) {
      // if there are more than 1 sub-task, we add several slices of the input
      // for each sub-task
      launcher.reserve_inputs(projections.size());
      for (auto&& projection : projections) {
        auto store_proj = input_partition->create_store_projection(launch_domain, projection);

        launcher.add_input(
          BaseArrayArg{RegionFieldArg{input_.get(), LEGION_READ_ONLY, std::move(store_proj)}});
      }
    } else {
      // otherwise we just add an entire input region to the task
      auto store_proj = input_partition->create_store_projection(launch_domain);

      launcher.add_input(
        BaseArrayArg{RegionFieldArg{input_.get(), LEGION_READ_ONLY, std::move(store_proj)}});
    }

    // calculating #of sub-tasks in the reduction task
    n_tasks = (n_tasks + radix_ - 1) / radix_;
    done    = n_tasks == 1;

    // adding output region
    auto field_space = runtime.create_field_space();
    auto field_id =
      runtime.allocate_field(field_space, RegionManager::FIELD_ID_BASE, input_->type()->size());

    // if this is not the last iteration of the while loop, we generate
    // a new output region
    if (n_tasks != 1) {
      new_output = runtime.create_store(input_->type(), 1);
      launcher.add_output(BaseArrayArg{OutputRegionArg{new_output.get(), field_space, field_id}});
    } else {
      launcher.add_output(BaseArrayArg{OutputRegionArg{output_.get(), field_space, field_id}});
    }

    // Every reduction task returns exactly one unbound store
    launcher.set_future_size(sizeof(std::size_t));

    launch_domain = Domain{DomainPoint{0}, DomainPoint{static_cast<coord_t>(n_tasks - 1)}};
    auto result   = launcher.execute(launch_domain);

    if (n_tasks != 1) {
      auto weighted = Weighted{result, launch_domain};

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

bool Reduce::needs_flush() const { return output_->needs_flush() || input_->needs_flush(); }

}  // namespace legate::detail
