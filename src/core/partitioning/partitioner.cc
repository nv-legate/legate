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
#include "core/data/logical_store.h"
#include "core/data/logical_store_detail.h"
#include "core/data/scalar.h"
#include "core/partitioning/constraint.h"
#include "core/partitioning/constraint_graph.h"
#include "core/partitioning/partition.h"
#include "core/runtime/launcher.h"
#include "core/runtime/operation.h"
#include "core/runtime/runtime.h"

namespace legate {

////////////////////////////////////////////////////
// legate::LaunchDomainResolver
////////////////////////////////////////////////////

class LaunchDomainResolver {
 private:
  static constexpr int32_t UNSET = -1;

 public:
  void record_launch_domain(const Legion::Domain& launch_domain);
  void record_unbound_store(int32_t unbound_dim);

 public:
  std::unique_ptr<Legion::Domain> resolve_launch_domain() const;

 private:
  bool must_be_sequential_{false};
  bool must_be_1d_{false};
  int32_t unbound_dim_{UNSET};
  std::set<Legion::Domain> launch_domains_;
  std::set<int64_t> launch_volumes_;
};

void LaunchDomainResolver::record_launch_domain(const Legion::Domain& launch_domain)
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

std::unique_ptr<Legion::Domain> LaunchDomainResolver::resolve_launch_domain() const
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
      return std::make_unique<Legion::Domain>(0, volume - 1);
    }
  } else {
#ifdef DEBUG_LEGATE
    assert(launch_domains_.size() == 1);
#endif
    auto& launch_domain = *launch_domains_.begin();
    if (unbound_dim_ != UNSET && launch_domain.dim != unbound_dim_) {
      int64_t volume = *launch_volumes_.begin();
      return std::make_unique<Legion::Domain>(0, volume - 1);
    }
    return std::make_unique<Legion::Domain>(launch_domain);
  }
}

////////////////////////////////////////////////////
// legate::Strategy
////////////////////////////////////////////////////

Strategy::Strategy() {}

bool Strategy::parallel(const Operation* op) const
{
  auto finder = launch_domains_.find(op);
  return finder != launch_domains_.end() && finder->second != nullptr;
}

const Legion::Domain* Strategy::launch_domain(const Operation* op) const
{
  auto finder = launch_domains_.find(op);
  return finder != launch_domains_.end() ? finder->second.get() : nullptr;
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

const std::shared_ptr<Partition>& Strategy::operator[](const Variable* partition_symbol) const
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

void Strategy::compute_launch_domains()
{
  std::map<const Operation*, LaunchDomainResolver> domain_resolvers;

  for (auto& assignment : assignments_) {
    auto& part_symb = assignment.first;
    auto& partition = assignment.second;
    auto* op        = part_symb.operation();

    if (partition->has_launch_domain()) {
      domain_resolvers[op].record_launch_domain(partition->launch_domain());
      continue;
    }

    auto store = op->find_store(&part_symb);

    if (store->unbound()) domain_resolvers[op].record_unbound_store(store->dim());
  }

  for (auto& pair : domain_resolvers)
    launch_domains_[pair.first] = pair.second.resolve_launch_domain();
}

////////////////////////////////////////////////////
// legate::Partitioner
////////////////////////////////////////////////////

Partitioner::Partitioner(std::vector<Operation*>&& operations)
  : operations_(std::forward<std::vector<Operation*>>(operations))
{
}

std::unique_ptr<Strategy> Partitioner::solve()
{
  ConstraintGraph constraints;

  for (auto op : operations_) op->add_to_constraint_graph(constraints);

  constraints.compute_equivalence_classes();

#ifdef DEBUG_LEGATE
  constraints.dump();
#endif

  auto strategy = std::make_unique<Strategy>();

  // Copy the list of partition symbols as we will sort them inplace
  std::vector<const Variable*> partition_symbols(constraints.partition_symbols());

  solve_for_unbound_stores(partition_symbols, strategy.get(), constraints);

  std::stable_sort(partition_symbols.begin(),
                   partition_symbols.end(),
                   [](const auto& part_symb_a, const auto& part_symb_b) {
                     auto get_storage_size = [](const auto& part_symb) {
                       auto* op    = part_symb->operation();
                       auto* store = op->find_store(part_symb);
#ifdef DEBUG_LEGATE
                       assert(!store->unbound());
#endif
                       return store->storage_size();
                     };
                     return get_storage_size(part_symb_a) > get_storage_size(part_symb_b);
                   });

  for (auto& part_symb : partition_symbols) {
    if (strategy->has_assignment(part_symb)) continue;

    auto* op       = part_symb->operation();
    auto store     = op->find_store(part_symb);
    auto partition = store->find_or_create_key_partition();

    std::vector<const Variable*> equiv_class;
    constraints.find_equivalence_class(part_symb, equiv_class);

    for (auto symb : equiv_class) strategy->insert(symb, partition);
  }

  strategy->compute_launch_domains();

  return std::move(strategy);
}

void Partitioner::solve_for_unbound_stores(std::vector<const Variable*>& partition_symbols,
                                           Strategy* strategy,
                                           const ConstraintGraph& constraints)
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

    std::vector<const Variable*> equiv_class;
    constraints.find_equivalence_class(part_symb, equiv_class);
    std::shared_ptr<Partition> partition(create_no_partition());
    auto field_space = runtime->create_field_space();

    for (auto symb : equiv_class) strategy->insert(symb, partition, field_space);
  }

  partition_symbols.swap(filtered);
}

}  // namespace legate
