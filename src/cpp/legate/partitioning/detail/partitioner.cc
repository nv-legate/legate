/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/partitioner.h>

#include <legate/data/detail/logical_store.h>
#include <legate/data/detail/transform/transform_stack.h>
#include <legate/operation/detail/operation.h>
#include <legate/partitioning/detail/constraint_solver.h>
#include <legate/partitioning/detail/launch_domain_resolver.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/partitioning/detail/partition/no_partition.h>
#include <legate/partitioning/detail/partition/opaque.h>
#include <legate/runtime/detail/projection.h>
#include <legate/runtime/detail/region_manager.h>
#include <legate/runtime/detail/runtime.h>

#include <algorithm>
#include <tuple>
#include <utility>
#include <vector>

namespace legate::detail {

namespace {

/*
 * We filter all unbounded stores and for each of them create a placeholder
 * Opaque partition. The partition will be filled with needed information like
 * the IndexSpace, launch domain etc after the task using the unbounded store
 * is submitted to Legion in post_process_unbounded_store.
 */
std::vector<const Variable*> create_fields_for_unbound_stores(const ConstraintSolver& solver,
                                                              Strategy* strategy)
{
  auto&& runtime = Runtime::get_runtime();
  auto ret       = std::vector<const Variable*>{};

  ret.reserve(solver.partition_symbols().size());

  for (auto&& equiv_class : solver.equivalence_classes()) {
    auto&& partition_symbols = equiv_class.partition_symbols;

    if (!equiv_class.IS_UNBOUND) {
      std::copy(partition_symbols.begin(), partition_symbols.end(), std::back_inserter(ret));
      continue;
    }

    auto field_space   = runtime.create_field_space();
    auto next_field_id = RegionManager::FIELD_ID_BASE;

    for (const auto* symb : partition_symbols) {
      if (next_field_id - RegionManager::FIELD_ID_BASE >= RegionManager::MAX_NUM_FIELDS) {
        field_space   = runtime.create_field_space();
        next_field_id = RegionManager::FIELD_ID_BASE;
      }

      const auto field_id =
        runtime.allocate_field(field_space, next_field_id++, symb->store()->type()->size());

      strategy->insert(*symb, field_space, field_id);
    }
  }

  return ret;
}

void create_partition_for_bound_stores(const Operation* op,
                                       std::vector<const Variable*> remaining_symbols,
                                       const ConstraintSolver& solver,
                                       Strategy* strategy)
{
  const auto comparison_key = [&solver, &op](const Variable& part_symb) {
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
    if (strategy->has_assignment(*part_symb) || solver.is_dependent(*part_symb)) {
      continue;
    }

    const auto& equiv_class  = solver.find_equivalence_class(*part_symb);
    const auto& restrictions = solver.find_restrictions(*part_symb);

    auto partition = op->find_store(part_symb)->find_or_create_key_partition(
      op->machine(), op->parallel_policy(), restrictions);

    strategy->record_key_partition({}, *part_symb);
    LEGATE_ASSERT(partition != nullptr);
    for (const auto* symb : equiv_class) {
      strategy->insert(*symb, partition);
    }
  }

  solver.solve_dependent_constraints(strategy);
}

void resolve_launch_domain_and_store_projections(const Operation* operation,
                                                 const ConstraintSolver& solver,
                                                 Strategy* strategy)
{
  LaunchDomainResolver domain_resolver{};

  for (auto&& equiv_class : solver.equivalence_classes()) {
    const auto* part_symb = equiv_class.partition_symbols.front();

    if (auto&& store = operation->find_store(part_symb); store->deferred_bound()) {
      domain_resolver.record_unbound_store(store->dim());
    } else if (auto&& partition = (*strategy)[*part_symb]; partition->has_launch_domain()) {
      domain_resolver.record_launch_domain(partition->launch_domain());
    } else if (!operation->supports_replicated_write() && solver.is_output(*part_symb)) {
      domain_resolver.set_must_be_sequential(true);
    }
  }

  const auto launch_domain = domain_resolver.resolve_launch_domain();

  strategy->set_launch_domain(launch_domain);

  // If this is a sequential launch, store projections are all identity
  if (!launch_domain.is_valid()) {
    return;
  }

  for (auto&& equiv_class : solver.equivalence_classes()) {
    if (equiv_class.IS_UNBOUND) {
      continue;
    }

    const auto& partition = (*strategy)[*equiv_class.partition_symbols.front()];

    if (!partition->has_color_shape()) {
      continue;
    }

    const auto& color_shape = partition->color_shape();

    for (auto&& part_symb : equiv_class.partition_symbols) {
      auto&& store = operation->find_store(part_symb);

      if (store->has_scalar_storage()) {
        continue;
      }
      strategy->insert_store_projection(
        Strategy::PrivateKey{},
        *part_symb,
        Partitioner::infer_store_projection(launch_domain, color_shape, store->transform()));
    }
  }
}

void create_placeholder_partitions_for_unbound_stores(const Operation* op,
                                                      const ConstraintSolver& solver)
{
  for (auto&& equiv_class : solver.equivalence_classes()) {
    if (!equiv_class.IS_UNBOUND) {
      continue;
    }

    // Create placeholder Opaque partition
    const InternalSharedPtr<Opaque> opaque_partition = create_opaque();

    for (const auto* part_symb : equiv_class.partition_symbols) {
      op->find_store(part_symb)->set_key_partition(
        op->machine(), op->parallel_policy(), opaque_partition);
    }
  }
}

}  // namespace

Strategy Partitioner::partition_stores(Operation* op)
{
  ConstraintSolver solver;

  op->add_to_solver(solver);

  solver.solve_constraints();

  solver.dump();

  auto strategy = Strategy{op};

  // Copy the list of partition symbols as we will sort them inplace
  auto remaining_symbols = create_fields_for_unbound_stores(solver, &strategy);

  create_partition_for_bound_stores(op, std::move(remaining_symbols), solver, &strategy);

  resolve_launch_domain_and_store_projections(op, solver, &strategy);

  if (strategy.launch_domain().is_valid()) {
    create_placeholder_partitions_for_unbound_stores(op, solver);
  }

  strategy.dump();

  return strategy;
}

Legion::ProjectionID Partitioner::infer_store_projection(
  const Domain& launch_domain,
  Span<const std::uint64_t> color_shape,
  const InternalSharedPtr<TransformStack>& transform)
{
  const auto ndim = color_shape.size();

  // Easy case where the store and launch domain have the same number of dimensions
  if (static_cast<std::size_t>(launch_domain.dim) == ndim) {
    return transform->identity() ? 0
                                 : Runtime::get_runtime().get_affine_projection(
                                     ndim, transform->invert(proj::create_symbolic_point(ndim)));
  }

  // If we're here, it means the launch domain has to be 1D due to mixed store dimensionalities
  if (launch_domain.dim != 1) {
    // This can happen only with ManualTasks, as AutoTasks don't get a multi-dimensional launch
    // domain when stores have different numbers of dimensions
    throw detail::TracedException<std::runtime_error>{
      "The launch domain must be 1D if the color shape has a different number of dimensions from "
      "the launch domain"};
  }

  // Check if the color shape has only one dimension of a non-unit extent, in which case
  // delinearization would simply be projections
  if (std::count_if(
        color_shape.begin(), color_shape.end(), [](const auto& ext) { return ext != 1; }) == 1) {
    SymbolicPoint embedding;

    embedding.reserve(color_shape.size());
    for (auto&& ext : color_shape) {
      embedding.append_inplace(ext != 1 ? dimension(0) : constant(0));
    }

    return Runtime::get_runtime().get_affine_projection(launch_domain.dim,
                                                        transform->invert(std::move(embedding)));
  }

  // When the store wasn't transformed, we could simply return the top-level delinearizing functor
  return transform->identity()
           ? Runtime::get_runtime().get_delinearizing_projection(color_shape)
           : Runtime::get_runtime().get_compound_projection(
               color_shape, transform->invert(proj::create_symbolic_point(ndim)));
}

}  // namespace legate::detail
