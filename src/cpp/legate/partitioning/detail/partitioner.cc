/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/partitioner.h>

#include <legate/data/detail/logical_store.h>
#include <legate/data/detail/transform/transform_stack.h>
#include <legate/operation/detail/operation.h>
#include <legate/operation/projection.h>
#include <legate/partitioning/detail/constraint_solver.h>
#include <legate/partitioning/detail/launch_domain_resolver.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/partitioning/detail/partition/no_partition.h>
#include <legate/partitioning/detail/partition/opaque.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/detail/small_vector.h>

#include <algorithm>
#include <tuple>
#include <utility>
#include <vector>

namespace legate::detail {

namespace {

void create_partition_for_bound_stores(const Operation* op,
                                       const ConstraintSolver& solver,
                                       Strategy* strategy)
{
  std::vector<const Variable*> symbols_for_bound_stores{};

  symbols_for_bound_stores.reserve(solver.partition_symbols().size());

  for (auto&& equiv_class : solver.equivalence_classes()) {
    if (equiv_class.IS_UNBOUND) {
      continue;
    }
    // A bound store: leave it for key-partition creation.
    symbols_for_bound_stores.insert(symbols_for_bound_stores.end(),
                                    equiv_class.partition_symbols.begin(),
                                    equiv_class.partition_symbols.end());
  }

  const auto comparison_key = [&solver, &op](const Variable& part_symb) {
    auto&& store            = op->find_store(&part_symb);
    const auto has_key_part = store->has_key_partition(
      op->machine(), op->parallel_policy(), solver.find_restrictions(part_symb));

    LEGATE_ASSERT(!store->unbound());
    return std::make_tuple(store->storage_size(), has_key_part, solver.find_access_mode(part_symb));
  };

  std::stable_sort(symbols_for_bound_stores.begin(),
                   symbols_for_bound_stores.end(),
                   [&comparison_key](const Variable* part_symb_a, const Variable* part_symb_b) {
                     return comparison_key(*part_symb_a) > comparison_key(*part_symb_b);
                   });

  for (const auto* part_symb : symbols_for_bound_stores) {
    if (strategy->has_assignment(*part_symb) || solver.is_dependent(*part_symb)) {
      continue;
    }

    const auto& equiv_class  = solver.find_equivalence_class(*part_symb);
    const auto& restrictions = solver.find_restrictions(*part_symb);

    auto partition = op->find_store(part_symb)->find_or_create_key_partition(
      op->machine(), op->parallel_policy(), restrictions);
    LEGATE_ASSERT(partition != nullptr);

    strategy->record_key_partition({}, *part_symb);
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

    if (equiv_class.IS_UNBOUND) {
      // We only consider "active" dimensions of a store when resolving the launch domain so that we
      // can allow unbound stores to have different numbers of dimensions from bound stores.
      domain_resolver.record_unbound_store(operation->find_store(part_symb)->dim() -
                                           equiv_class.restrictions.count_restricted());
    } else if (auto&& partition = (*strategy)[*part_symb]; partition->has_color_shape()) {
      domain_resolver.record_launch_domain(
        equiv_class.has_restrictions
          ? equiv_class.restrictions.prune_dimensions(partition->launch_domain())
          : partition->launch_domain());
    } else if (!operation->supports_replicated_write() && solver.is_output(*part_symb)) {
      domain_resolver.set_must_be_sequential(true);
    }
  }

  const auto launch_domain = domain_resolver.resolve_launch_domain();

  strategy->set_launch_domain(launch_domain);

  // If this is a sequential launch, store projections are all identity
  if (launch_domain.dim() == 0) {
    return;
  }

  for (auto&& equiv_class : solver.equivalence_classes()) {
    auto projection = equiv_class.restrictions.to_projection();

    if (equiv_class.IS_UNBOUND) {
      auto partition = create_opaque(equiv_class.restrictions.embed_to_color_shape(launch_domain));

      for (auto&& part_symb : equiv_class.partition_symbols) {
        strategy->insert(*part_symb, partition);
        strategy->insert_store_projection(Strategy::PrivateKey{}, *part_symb, projection);
      }
      continue;
    }

    const auto& partition = (*strategy)[*equiv_class.partition_symbols.front()];

    // Unbound output stores get an identity projection: each launch point maps to the color of the
    // piece it produces. The placeholder Opaque partition has a color shape, so this falls through
    // to the generic projection logic below (its launch domain is 1D, so `infer_store_projection`
    // yields the identity).
    if (!partition->has_color_shape()) {
      continue;
    }

    const auto& color_shape = partition->color_shape();

    for (auto&& part_symb : equiv_class.partition_symbols) {
      auto&& store = operation->find_store(part_symb);

      if (equiv_class.has_restrictions && launch_domain.dim() > 1) {
        // When the launch domain has more than one dimension, store projections can never be
        // delinearizing projections
        strategy->insert_store_projection(
          Strategy::PrivateKey{}, *part_symb, store->transform()->invert(projection));
        continue;
      }

      strategy->insert_store_projection(
        Strategy::PrivateKey{},
        *part_symb,
        Partitioner::infer_store_projection(launch_domain, color_shape, store->transform()));
    }
  }
}

}  // namespace

Strategy Partitioner::partition_stores(Operation* op)
{
  if (Runtime::get_runtime().single_process()) {
    return Strategy{op};
  }

  ConstraintSolver solver;

  op->add_to_solver(solver);

  solver.solve_constraints();

  solver.dump();

  auto strategy = Strategy{op};

  create_partition_for_bound_stores(op, solver, &strategy);

  resolve_launch_domain_and_store_projections(op, solver, &strategy);

  strategy.dump();

  return strategy;
}

namespace {

SymbolicPoint create_symbolic_point(std::uint32_t ndim)
{
  std::vector<SymbolicExpr> exprs;

  exprs.reserve(ndim);
  for (std::uint32_t dim = 0; dim < ndim; ++dim) {
    exprs.emplace_back(dim);
  }
  return SymbolicPoint{std::move(exprs)};
}

}  // namespace

SymbolicPoint Partitioner::infer_store_projection(
  const Rect& launch_domain,
  const Point& color_shape,
  const InternalSharedPtr<TransformStack>& transform)
{
  const auto ndim = color_shape.size();

  // Easy case where the store and launch domain have the same number of dimensions
  if (static_cast<std::size_t>(launch_domain.dim()) == ndim) {
    return transform->identity() ? SymbolicPoint{} : transform->invert(create_symbolic_point(ndim));
  }

  // If we're here, it means the launch domain has to be 1D due to mixed store dimensionalities
  if (launch_domain.dim() != 1) {
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

    return transform->invert(std::move(embedding));
  }

  LEGATE_ABORT("Not supported yet");

  return {};
}

}  // namespace legate::detail
