/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/scatter_gather.h>

#include <legate/operation/detail/copy_launcher.h>
#include <legate/partitioning/detail/constraint.h>
#include <legate/partitioning/detail/constraint_solver.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/partitioning/detail/partitioner.h>
#include <legate/utilities/detail/traced_exception.h>

#include <fmt/format.h>

#include <stdexcept>

namespace legate::detail {

ScatterGather::ScatterGather(InternalSharedPtr<LogicalStore> target,
                             InternalSharedPtr<LogicalStore> target_indirect,
                             InternalSharedPtr<LogicalStore> source,
                             InternalSharedPtr<LogicalStore> source_indirect,
                             std::uint64_t unique_id,
                             std::int32_t priority,
                             mapping::detail::Machine machine,
                             std::optional<std::int32_t> redop_kind)
  : Operation{unique_id, priority, std::move(machine)},
    target_{target, declare_partition()},
    target_indirect_{target_indirect, declare_partition()},
    source_{source, declare_partition()},
    source_indirect_{source_indirect, declare_partition()},
    constraint_{align(source_indirect_.variable, target_indirect_.variable)},
    redop_kind_{redop_kind}
{
  record_partition_(
    target_.variable, std::move(target), redop_kind_ ? AccessMode::REDUCE : AccessMode::WRITE);
  record_partition_(target_indirect_.variable, std::move(target_indirect), AccessMode::READ);
  record_partition_(source_.variable, std::move(source), AccessMode::READ);
  record_partition_(source_indirect_.variable, std::move(source_indirect), AccessMode::READ);
}

void ScatterGather::validate()
{
  if (*source_.store->type() != *target_.store->type()) {
    throw TracedException<std::invalid_argument>{"Source and targets must have the same type"};
  }
  constexpr auto validate_store = [](const auto& store) {
    if (store->unbound() || store->has_scalar_storage() || store->transformed()) {
      throw TracedException<std::invalid_argument>{
        "ScatterGather accepts only normal, untransformed, region-backed stores"};
    }
  };
  validate_store(target_.store);
  validate_store(target_indirect_.store);
  validate_store(source_.store);
  validate_store(source_indirect_.store);

  if (!is_point_type(source_indirect_.store->type(), source_.store->dim())) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Source indirection store should contain {}-D points", source_.store->dim())};
  }
  if (!is_point_type(target_indirect_.store->type(), target_.store->dim())) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Target indirection store should contain {}-D points", target_.store->dim())};
  }

  constraint_->validate();
}

void ScatterGather::launch(Strategy* p_strategy)
{
  auto& strategy       = *p_strategy;
  auto launcher        = CopyLauncher{machine_, priority()};
  auto&& launch_domain = strategy.launch_domain(*this);

  launcher.add_input(source_.store, create_store_projection_(strategy, launch_domain, source_));
  launcher.add_source_indirect(source_indirect_.store,
                               create_store_projection_(strategy, launch_domain, source_indirect_));

  if (!redop_kind_) {
    launcher.add_inout(target_.store, create_store_projection_(strategy, launch_domain, target_));
  } else {
    auto store_partition = create_store_partition(target_.store, strategy[*target_.variable]);
    auto proj            = store_partition->create_store_projection(launch_domain);

    proj.set_reduction_op(target_.store->type()->find_reduction_operator(redop_kind_.value()));
    launcher.add_reduction(target_.store, std::move(proj));
  }

  launcher.add_target_indirect(target_indirect_.store,
                               create_store_projection_(strategy, launch_domain, target_indirect_));
  launcher.set_target_indirect_out_of_range(target_indirect_out_of_range_);
  launcher.set_source_indirect_out_of_range(source_indirect_out_of_range_);

  if (launch_domain.is_valid()) {
    launcher.execute(launch_domain);
  } else {
    launcher.execute_single();
  }
}

void ScatterGather::add_to_solver(ConstraintSolver& solver)
{
  // TODO(amberhassaan): do not move constraints until
  // https://github.com/nv-legate/legate.internal/issues/3120 is resolved
  // solver.add_constraint(std::move(constraint_));
  solver.add_constraint(constraint_);
  solver.add_partition_symbol(target_.variable,
                              redop_kind_ ? AccessMode::REDUCE : AccessMode::WRITE);
  solver.add_partition_symbol(target_indirect_.variable, AccessMode::READ);
  solver.add_partition_symbol(source_.variable, AccessMode::READ);
  solver.add_partition_symbol(source_indirect_.variable, AccessMode::READ);
}

bool ScatterGather::needs_flush() const
{
  return target_.needs_flush() || source_.needs_flush() || target_indirect_.needs_flush() ||
         source_indirect_.needs_flush();
}

}  // namespace legate::detail
