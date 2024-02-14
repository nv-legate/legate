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

#include "core/operation/detail/scatter.h"

#include "core/operation/detail/copy_launcher.h"
#include "core/partitioning/detail/constraint.h"
#include "core/partitioning/detail/constraint_solver.h"
#include "core/partitioning/detail/partitioner.h"
#include "core/partitioning/partition.h"

namespace legate::detail {

Scatter::Scatter(InternalSharedPtr<LogicalStore> target,
                 InternalSharedPtr<LogicalStore> target_indirect,
                 InternalSharedPtr<LogicalStore> source,
                 std::uint64_t unique_id,
                 mapping::detail::Machine&& machine,
                 std::optional<std::int32_t> redop)
  : Operation{unique_id, std::move(machine)},
    target_{target, declare_partition()},
    target_indirect_{target_indirect, declare_partition()},
    source_{source, declare_partition()},
    constraint_(align(source_.variable, target_indirect_.variable)),
    redop_{redop}
{
  record_partition(target_.variable, std::move(target));
  record_partition(target_indirect_.variable, std::move(target_indirect));
  record_partition(source_.variable, std::move(source));
}

void Scatter::validate()
{
  if (*source_.store->type() != *target_.store->type()) {
    throw std::invalid_argument("Source and targets must have the same type");
  }
  auto validate_store = [](const auto& store) {
    if (store->unbound() || store->has_scalar_storage() || store->transformed()) {
      throw std::invalid_argument(
        "Scatter accepts only normal, untransformed, region-backed stores");
    }
  };
  validate_store(target_.store);
  validate_store(target_indirect_.store);
  validate_store(source_.store);

  if (!is_point_type(target_indirect_.store->type(), target_.store->dim())) {
    throw std::invalid_argument("Indirection store should contain " +
                                std::to_string(target_.store->dim()) + "-D points");
  }

  constraint_->validate();
}

void Scatter::launch(Strategy* p_strategy)
{
  auto& strategy     = *p_strategy;
  auto launcher      = CopyLauncher{machine_};
  auto launch_domain = strategy.launch_domain(this);

  launcher.add_input(source_.store, create_store_projection(strategy, launch_domain, source_));

  if (!redop_) {
    launcher.add_inout(target_.store, create_store_projection(strategy, launch_domain, target_));
  } else {
    auto store_partition = create_store_partition(target_.store, strategy[target_.variable]);
    auto proj            = store_partition->create_store_projection(launch_domain);

    proj->set_reduction_op(static_cast<Legion::ReductionOpID>(
      target_.store->type()->find_reduction_operator(redop_.value())));
    launcher.add_reduction(target_.store, std::move(proj));
  }

  launcher.add_target_indirect(target_indirect_.store,
                               create_store_projection(strategy, launch_domain, target_indirect_));
  launcher.set_target_indirect_out_of_range(out_of_range_);

  if (launch_domain.is_valid()) {
    return launcher.execute(launch_domain);
  }
  return launcher.execute_single();
}

void Scatter::add_to_solver(ConstraintSolver& solver)
{
  solver.add_constraint(std::move(constraint_));
  solver.add_partition_symbol(target_.variable, !redop_ ? AccessMode::WRITE : AccessMode::REDUCE);
  solver.add_partition_symbol(target_indirect_.variable, AccessMode::READ);
  solver.add_partition_symbol(source_.variable, AccessMode::READ);
}

std::string Scatter::to_string() const { return "Scatter:" + std::to_string(unique_id_); }

}  // namespace legate::detail
