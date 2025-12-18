/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/partitioner.h>

#include <legate/data/detail/logical_store.h>
#include <legate/operation/detail/operation.h>
#include <legate/partitioning/detail/constraint_solver.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/partitioning/detail/partition/no_partition.h>
#include <legate/runtime/detail/region_manager.h>
#include <legate/runtime/detail/runtime.h>

#include <algorithm>
#include <tuple>
#include <utility>
#include <vector>

namespace legate::detail {

Strategy Partitioner::partition_stores()
{
  ConstraintSolver solver;

  for (auto&& op : operations_) {
    op->add_to_solver(solver);
  }

  solver.solve_constraints();

  solver.dump();

  auto strategy = Strategy{};

  // Copy the list of partition symbols as we will sort them inplace
  auto remaining_symbols = handle_unbound_stores_(solver.partition_symbols(), solver, &strategy);

  const auto comparison_key = [&solver](const Variable& part_symb) {
    const auto* const op    = part_symb.operation();
    auto&& store            = op->find_store(&part_symb);
    const auto has_key_part = store->has_key_partition(
      op->machine(), op->parallel_policy(), solver.find_restrictions(part_symb));

    LEGATE_ASSERT(!store->unbound());
    return std::make_tuple(store->storage_size(), has_key_part, solver.find_access_mode(part_symb));
  };

  std::stable_sort(remaining_symbols.begin(),
                   remaining_symbols.end(),
                   [&comparison_key](const Variable* part_symb_a, const Variable* part_symb_b) {
                     return comparison_key(*part_symb_a) > comparison_key(*part_symb_b);
                   });

  for (const auto* part_symb : remaining_symbols) {
    if (strategy.has_assignment(*part_symb) || solver.is_dependent(*part_symb)) {
      continue;
    }

    const auto& equiv_class  = solver.find_equivalence_class(*part_symb);
    const auto& restrictions = solver.find_restrictions(*part_symb);

    auto* op       = part_symb->operation();
    auto partition = op->find_store(part_symb)->find_or_create_key_partition(
      op->machine(), op->parallel_policy(), restrictions);

    strategy.record_key_partition({}, *part_symb);
    LEGATE_ASSERT(partition != nullptr);
    for (const auto* symb : equiv_class) {
      strategy.insert(*symb, partition);
    }
  }

  solver.solve_dependent_constraints(&strategy);
  strategy.compute_launch_domains({}, solver);
  strategy.dump();
  return strategy;
}

std::vector<const Variable*> Partitioner::handle_unbound_stores_(
  Span<const Variable* const> partition_symbols, const ConstraintSolver& solver, Strategy* strategy)
{
  auto&& runtime        = Runtime::get_runtime();
  auto is_unbound_store = [&](const Variable* var) {
    const auto& part_symb = *var;

    if (strategy->has_assignment(part_symb)) {
      return true;
    }

    if (!part_symb.operation()->find_store(&part_symb)->unbound()) {
      return false;
    }

    auto&& equiv_class = solver.find_equivalence_class(part_symb);
    const InternalSharedPtr<Partition> partition{create_no_partition()};
    auto field_space   = runtime.create_field_space();
    auto next_field_id = RegionManager::FIELD_ID_BASE;

    for (const auto* symb : equiv_class) {
      if (next_field_id - RegionManager::FIELD_ID_BASE >= RegionManager::MAX_NUM_FIELDS) {
        field_space   = runtime.create_field_space();
        next_field_id = RegionManager::FIELD_ID_BASE;
      }
      const auto field_id =
        runtime.allocate_field(field_space, next_field_id++, symb->store()->type()->size());

      strategy->insert(*symb, partition, field_space, field_id);
    }
    return true;
  };

  std::vector<const Variable*> ret;

  ret.reserve(partition_symbols.size());
  std::copy_if(partition_symbols.begin(),
               partition_symbols.end(),
               std::back_inserter(ret),
               std::not_fn(std::move(is_unbound_store)));
  return ret;
}

}  // namespace legate::detail
