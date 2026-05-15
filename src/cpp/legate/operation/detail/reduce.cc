/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
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
#include <legate/partitioning/detail/partition/opaque.h>
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
  if (output_->unbound()) {
    output_->get_storage()->set_bound(true /* bound */);
  }
}

void Reduce::launch(Strategy* p_strategy)
{
  auto& strategy = *p_strategy;
  // Copy is deliberate, launch_domain is updated in the loop
  auto launch_domain = strategy.launch_domain();

  if (!launch_domain.is_valid()) {
    launch_single_();
    return;
  }

  LEGATE_ASSERT(launch_domain.dim == 1);

  auto&& runtime = detail::Runtime::get_runtime();
  auto n_tasks   = launch_domain.get_volume();

  LEGATE_ASSERT(n_tasks > 1);

  auto input_part      = strategy[*input_part_];
  auto input_partition = create_store_partition(input_, input_part);

  // generating projection functions to use in tree_reduction task
  std::vector<Legion::ProjectionID> projection_ids;

  projection_ids.reserve(static_cast<std::size_t>(radix_));
  for (std::int32_t i = 0; i < radix_; i++) {
    projection_ids.push_back(
      runtime.get_affine_projection(/*src_ndim=*/1, {SymbolicExpr{/*dim=*/0, radix_, i}}));
  }

  InternalSharedPtr<LogicalStore> new_output;
  bool done = false;

  do {
    auto launcher =
      detail::TaskLauncher{library_,
                           machine_,
                           parallel_policy(),
                           provenance_,
                           task_id_,
                           Legion::ProjectionID{0},
                           static_cast<Legion::MappingTagID>(CoreMappingTag::TREE_REDUCE)};

    launcher.set_priority(priority());

    // if there are more than 1 sub-task, we add several slices of the input
    // for each sub-task
    launcher.reserve_inputs(projection_ids.size());
    for (auto&& projection_id : projection_ids) {
      auto store_proj = StoreProjection{
        input_partition->storage_partition()->get_legion_partition(), projection_id};

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
      new_output = runtime.create_store(input_->type(), /*dim=*/1);
      launcher.add_output(BaseArrayArg{OutputRegionArg{new_output.get(), field_space, field_id}});

      // Create placeholder Opaque partition
      const InternalSharedPtr<Opaque> opaque_partition = create_opaque();
      new_output->set_key_partition(machine_, parallel_policy_, opaque_partition);
      new_output->get_storage()->set_bound(true /* bound */);
    } else {
      launcher.add_output(BaseArrayArg{OutputRegionArg{output_.get(), field_space, field_id}});
    }

    launcher.execute(Domain{DomainPoint{0}, DomainPoint{static_cast<coord_t>(n_tasks - 1)}});

    if (n_tasks != 1) {
      const auto& key_partition = new_output->get_current_key_partition();
      LEGATE_ASSERT(key_partition.has_value());
      auto output_partition = create_store_partition(
        new_output, *key_partition);  // NOLINT(bugprone-unchecked-optional-access)
      input_          = new_output;
      input_partition = output_partition;
    }
  } while (!done);

  // The partitioner is not smart enough to recognize that the tree reduce op produces a single,
  // unpartitioned unbound store at the end, and thus blindly assigns a placeholder partition to
  // every unbound store as its key partition when the launch domain is valid. Therefore, we need to
  // reset the key partition so that there'd be no invalid partition dangling in the output store.
  output_->reset_key_partition(/*flush=*/false);
}

void Reduce::launch_single_()
{
  auto&& runtime = detail::Runtime::get_runtime();
  auto launcher =
    detail::TaskLauncher{library_,
                         machine_,
                         parallel_policy(),
                         provenance_,
                         task_id_,
                         Legion::ProjectionID{0},
                         static_cast<Legion::MappingTagID>(CoreMappingTag::TREE_REDUCE)};

  launcher.set_priority(priority());
  launcher.add_input(
    BaseArrayArg{RegionFieldArg{input_.get(), LEGION_READ_ONLY, StoreProjection{}}});

  // adding output region
  auto field_space = runtime.create_field_space();
  auto field_id =
    runtime.allocate_field(field_space, RegionManager::FIELD_ID_BASE, input_->type()->size());

  launcher.add_output(BaseArrayArg{OutputRegionArg{output_.get(), field_space, field_id}});

  launcher.execute_single();
}

void Reduce::add_to_solver(detail::ConstraintSolver& solver)
{
  solver.add_partition_symbol(output_part_, AccessMode::WRITE);
  solver.add_partition_symbol(input_part_, AccessMode::READ);
}

bool Reduce::needs_flush() const { return output_->needs_flush() || input_->needs_flush(); }

}  // namespace legate::detail
