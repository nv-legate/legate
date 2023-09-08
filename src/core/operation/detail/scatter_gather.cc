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

#include "core/operation/detail/scatter_gather.h"

#include "core/operation/detail/copy_launcher.h"
#include "core/partitioning/detail/constraint.h"
#include "core/partitioning/detail/constraint_solver.h"
#include "core/partitioning/detail/partitioner.h"
#include "core/partitioning/partition.h"

namespace legate::detail {

ScatterGather::ScatterGather(std::shared_ptr<LogicalStore> target,
                             std::shared_ptr<LogicalStore> target_indirect,
                             std::shared_ptr<LogicalStore> source,
                             std::shared_ptr<LogicalStore> source_indirect,
                             int64_t unique_id,
                             mapping::detail::Machine&& machine,
                             std::optional<int32_t> redop)
  : Operation(unique_id, std::move(machine)),
    target_{target.get(), declare_partition()},
    target_indirect_{target_indirect.get(), declare_partition()},
    source_{source.get(), declare_partition()},
    source_indirect_{source_indirect.get(), declare_partition()},
    constraint_(align(source_indirect_.variable, target_indirect_.variable)),
    redop_{redop}
{
  record_partition(target_.variable, std::move(target));
  record_partition(target_indirect_.variable, std::move(target_indirect));
  record_partition(source_.variable, std::move(source));
  record_partition(source_indirect_.variable, std::move(source_indirect));
}

void ScatterGather::set_source_indirect_out_of_range(bool flag)
{
  source_indirect_out_of_range_ = flag;
}

void ScatterGather::set_target_indirect_out_of_range(bool flag)
{
  target_indirect_out_of_range_ = flag;
}

void ScatterGather::validate()
{
  if (*source_.store->type() != *target_.store->type()) {
    throw std::invalid_argument("Source and targets must have the same type");
  }
  auto validate_store = [](auto* store) {
    if (store->unbound() || store->has_scalar_storage() || store->transformed()) {
      throw std::invalid_argument(
        "ScatterGather accepts only normal, untransformed, region-backed stores");
    }
  };
  validate_store(target_.store);
  validate_store(target_indirect_.store);
  validate_store(source_.store);
  validate_store(source_indirect_.store);

  if (!is_point_type(source_indirect_.store->type(), source_.store->dim())) {
    throw std::invalid_argument("Source indirection store should contain " +
                                std::to_string(source_.store->dim()) + "-D points");
  }
  if (!is_point_type(target_indirect_.store->type(), target_.store->dim())) {
    throw std::invalid_argument("Target indirection store should contain " +
                                std::to_string(target_.store->dim()) + "-D points");
  }

  constraint_->validate();
}

void ScatterGather::launch(Strategy* p_strategy)
{
  auto& strategy = *p_strategy;
  CopyLauncher launcher(machine_);
  auto launch_domain = strategy.launch_domain(this);

  launcher.add_input(source_.store, create_projection_info(strategy, launch_domain, source_));
  launcher.add_source_indirect(source_indirect_.store,
                               create_projection_info(strategy, launch_domain, source_indirect_));

  if (!redop_) {
    launcher.add_inout(target_.store, create_projection_info(strategy, launch_domain, target_));
  } else {
    auto store_partition = target_.store->create_partition(strategy[target_.variable]);
    auto proj            = store_partition->create_projection_info(launch_domain);
    proj->set_reduction_op(target_.store->type()->find_reduction_operator(redop_.value()));
    launcher.add_reduction(target_.store, std::move(proj));
  }

  launcher.add_target_indirect(target_indirect_.store,
                               create_projection_info(strategy, launch_domain, target_indirect_));
  launcher.set_target_indirect_out_of_range(target_indirect_out_of_range_);
  launcher.set_source_indirect_out_of_range(source_indirect_out_of_range_);

  if (launch_domain != nullptr) {
    return launcher.execute(*launch_domain);
  } else {
    return launcher.execute_single();
  }
}

void ScatterGather::add_to_solver(ConstraintSolver& solver)
{
  solver.add_constraint(std::move(constraint_));
  solver.add_partition_symbol(target_.variable);
  solver.add_partition_symbol(target_indirect_.variable);
  solver.add_partition_symbol(source_.variable);
  solver.add_partition_symbol(source_indirect_.variable);
}

std::string ScatterGather::to_string() const
{
  return "ScatterGather:" + std::to_string(unique_id_);
}

}  // namespace legate::detail
