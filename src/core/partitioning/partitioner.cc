/* Copyright 2021 NVIDIA Corporation
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

#include "core/partitioning/partitioner.h"
#include "core/data/detail/logical_store.h"
#include "core/data/logical_store.h"
#include "core/data/scalar.h"
#include "core/operation/detail/operation.h"
#include "core/partitioning/constraint.h"
#include "core/partitioning/constraint_solver.h"
#include "core/partitioning/partition.h"
#include "core/runtime/detail/runtime.h"

namespace legate::detail {

////////////////////////////////////////////////////
// legate::detail::LaunchDomainResolver
////////////////////////////////////////////////////

class LaunchDomainResolver {
 private:
  static constexpr int32_t UNSET = -1;

 public:
  void record_launch_domain(const Domain& launch_domain);
  void record_unbound_store(int32_t unbound_dim);
  void set_must_be_sequential(bool must_be_sequential) { must_be_sequential_ = must_be_sequential; }

 public:
  std::unique_ptr<Domain> resolve_launch_domain() const;

 private:
  bool must_be_sequential_{false};
  bool must_be_1d_{false};
  int32_t unbound_dim_{UNSET};
  std::set<Domain> launch_domains_;
  std::set<int64_t> launch_volumes_;
};

void LaunchDomainResolver::record_launch_domain(const Domain& launch_domain)
{
  launch_domains_.insert(launch_domain);
  launch_volumes_.insert(launch_domain.get_volume());
  if (launch_domains_.size() > 1) must_be_1d_ = true;
  if (launch_volumes_.size() > 1) must_be_sequential_ = true;
}

void LaunchDomainResolver::record_unbound_store(int32_t unbound_dim)
{
  if (unbound_dim_ != UNSET && unbound_dim_ != unbound_dim)
    must_be_sequential_ = true;
  else
    unbound_dim_ = unbound_dim;
}

std::unique_ptr<Domain> LaunchDomainResolver::resolve_launch_domain() const
{
  if (must_be_sequential_ || launch_domains_.empty()) return nullptr;
  if (must_be_1d_) {
    if (unbound_dim_ != UNSET && unbound_dim_ > 1)
      return nullptr;
    else {
#ifdef DEBUG_LEGATE
      assert(launch_volumes_.size() == 1);
#endif
      int64_t volume = *launch_volumes_.begin();
      return std::make_unique<Domain>(0, volume - 1);
    }
  } else {
#ifdef DEBUG_LEGATE
    assert(launch_domains_.size() == 1);
#endif
    auto& launch_domain = *launch_domains_.begin();
    if (unbound_dim_ != UNSET && launch_domain.dim != unbound_dim_) {
      int64_t volume = *launch_volumes_.begin();
      return std::make_unique<Domain>(0, volume - 1);
    }
    return std::make_unique<Domain>(launch_domain);
  }
}

////////////////////////////////////////////////////
// legate::detail::Strategy
////////////////////////////////////////////////////

Strategy::Strategy() {}

bool Strategy::parallel(const Operation* op) const
{
  auto finder = launch_domains_.find(op);
  return finder != launch_domains_.end() && finder->second != nullptr;
}

const Domain* Strategy::launch_domain(const Operation* op) const
{
  auto finder = launch_domains_.find(op);
  return finder != launch_domains_.end() ? finder->second.get() : nullptr;
}

void Strategy::set_launch_shape(const Operation* op, const Shape& shape)
{
#ifdef DEBUG_LEGATE
  assert(launch_domains_.find(op) == launch_domains_.end());
#endif
  launch_domains_.insert({op, std::make_unique<Domain>(to_domain(shape))});
}

void Strategy::insert(const Variable* partition_symbol, std::shared_ptr<Partition> partition)
{
#ifdef DEBUG_LEGATE
  assert(assignments_.find(*partition_symbol) == assignments_.end());
#endif
  assignments_.insert({*partition_symbol, std::move(partition)});
}

void Strategy::insert(const Variable* partition_symbol,
                      std::shared_ptr<Partition> partition,
                      Legion::FieldSpace field_space)
{
#ifdef DEBUG_LEGATE
  assert(field_spaces_.find(*partition_symbol) == field_spaces_.end());
#endif
  field_spaces_.insert({*partition_symbol, field_space});
  insert(partition_symbol, std::move(partition));
}

bool Strategy::has_assignment(const Variable* partition_symbol) const
{
  return assignments_.find(*partition_symbol) != assignments_.end();
}

std::shared_ptr<Partition> Strategy::operator[](const Variable* partition_symbol) const
{
  auto finder = assignments_.find(*partition_symbol);
#ifdef DEBUG_LEGATE
  assert(finder != assignments_.end());
#endif
  return finder->second;
}

const Legion::FieldSpace& Strategy::find_field_space(const Variable* partition_symbol) const
{
  auto finder = field_spaces_.find(*partition_symbol);
#ifdef DEBUG_LEGATE
  assert(finder != field_spaces_.end());
#endif
  return finder->second;
}

bool Strategy::is_key_partition(const Variable* partition_symbol) const
{
  return key_partition_.has_value() && key_partition_.value() == partition_symbol;
}

void Strategy::dump() const
{
  log_legate.debug("===== Solution =====");
  for (const auto& [symbol, part] : assignments_)
    log_legate.debug() << symbol.to_string() << ": " << part->to_string();
  for (const auto& [symbol, fspace] : field_spaces_)
    log_legate.debug() << symbol.to_string() << ": " << fspace;
  for (const auto& [op, domain] : launch_domains_) {
    if (nullptr == domain)
      log_legate.debug() << op->to_string() << ": (sequential)";
    else
      log_legate.debug() << op->to_string() << ": " << *domain;
  }
  log_legate.debug("====================");
}

void Strategy::compute_launch_domains(const ConstraintSolver& solver)
{
  std::map<const Operation*, LaunchDomainResolver> domain_resolvers;

  for (auto& [part_symb, partition] : assignments_) {
    auto* op              = part_symb.operation();
    auto& domain_resolver = domain_resolvers[op];

    if (partition->has_launch_domain()) {
      domain_resolver.record_launch_domain(partition->launch_domain());
      continue;
    }

    auto store = op->find_store(&part_symb);
    if (store->unbound())
      domain_resolver.record_unbound_store(store->dim());
    else if (solver.is_output(part_symb))
      domain_resolver.set_must_be_sequential(true);
  }

  for (auto& [op, domain_resolver] : domain_resolvers)
    launch_domains_[op] = domain_resolver.resolve_launch_domain();
}

void Strategy::record_key_partition(const Variable* partition_symbol)
{
  if (!key_partition_) key_partition_ = partition_symbol;
}

////////////////////////////////////////////////////
// legate::detail::Partitioner
////////////////////////////////////////////////////

Partitioner::Partitioner(std::vector<Operation*>&& operations) : operations_(std::move(operations))
{
}

std::unique_ptr<Strategy> Partitioner::partition_stores()
{
  ConstraintSolver solver;

  for (auto op : operations_) op->add_to_solver(solver);

  solver.solve_constraints();

#ifdef DEBUG_LEGATE
  solver.dump();
#endif

  auto strategy = std::make_unique<Strategy>();

  // Copy the list of partition symbols as we will sort them inplace
  auto partition_symbols = solver.partition_symbols();

  auto remaining_symbols = handle_unbound_stores(strategy.get(), partition_symbols, solver);

  auto comparison_key = [&solver](const auto& part_symb) {
    auto* op   = part_symb->operation();
    auto store = op->find_store(part_symb);
    auto has_key_part =
      store->has_key_partition(op->machine(), solver.find_restrictions(part_symb));
#ifdef DEBUG_LEGATE
    assert(!store->unbound());
#endif
    return std::make_pair(store->storage_size(), has_key_part);
  };

  std::stable_sort(remaining_symbols.begin(),
                   remaining_symbols.end(),
                   [&solver, &comparison_key](const auto& part_symb_a, const auto& part_symb_b) {
                     return comparison_key(part_symb_a) > comparison_key(part_symb_b);
                   });

  for (auto& part_symb : remaining_symbols) {
    if (strategy->has_assignment(part_symb))
      continue;
    else if (solver.is_dependent(*part_symb))
      continue;

    const auto& equiv_class  = solver.find_equivalence_class(part_symb);
    const auto& restrictions = solver.find_restrictions(part_symb);

    auto* op       = part_symb->operation();
    auto store     = op->find_store(part_symb);
    auto partition = store->find_or_create_key_partition(op->machine(), restrictions);
    strategy->record_key_partition(part_symb);
#ifdef DEBUG_LEGATE
    assert(partition != nullptr);
#endif

    for (auto symb : equiv_class) strategy->insert(symb, partition);
  }

  solver.solve_dependent_constraints(*strategy);

  strategy->compute_launch_domains(solver);

#ifdef DEBUG_LEGATE
  strategy->dump();
#endif

  return strategy;
}

std::vector<const Variable*> Partitioner::handle_unbound_stores(
  Strategy* strategy,
  const std::vector<const Variable*>& partition_symbols,
  const ConstraintSolver& solver)
{
  auto runtime = Runtime::get_runtime();

  std::vector<const Variable*> filtered;
  filtered.reserve(partition_symbols.size());

  for (auto* part_symb : partition_symbols) {
    auto* op   = part_symb->operation();
    auto store = op->find_store(part_symb);

    if (!store->unbound()) {
      filtered.push_back(part_symb);
      continue;
    }

    auto equiv_class = solver.find_equivalence_class(part_symb);
    std::shared_ptr<Partition> partition(create_no_partition());
    auto field_space = runtime->create_field_space();

    for (auto symb : equiv_class) strategy->insert(symb, partition, field_space);
  }

  return filtered;
}

}  // namespace legate::detail
