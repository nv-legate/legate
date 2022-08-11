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
// legate::Strategy
////////////////////////////////////////////////////

Strategy::Strategy() {}

bool Strategy::parallel(const Operation* op) const
{
  auto finder = launch_domains_.find(op);
  return (launch_domains_.end() != finder) && (nullptr != finder->second);
}

bool Strategy::has_launch_domain(const Operation* op) const
{
  return launch_domains_.find(op) != launch_domains_.end();
}

Legion::Domain Strategy::launch_domain(const Operation* op) const
{
  auto finder = launch_domains_.find(op);
  assert(finder != launch_domains_.end());
  return *finder->second;
}

void Strategy::set_single_launch(const Operation* op) { launch_domains_[op] = nullptr; }

void Strategy::set_launch_domain(const Operation* op, const Legion::Domain& launch_domain)
{
  launch_domains_[op] = std::make_unique<Legion::Domain>(launch_domain);
}

void Strategy::insert(const Variable* partition_symbol, std::shared_ptr<Partition> partition)
{
#ifdef DEBUG_LEGATE
  assert(assignments_.find(*partition_symbol) == assignments_.end());
#endif
  assignments_.insert({*partition_symbol, std::move(partition)});
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
  std::stable_sort(partition_symbols.begin(),
                   partition_symbols.end(),
                   [](const auto& part_symb_a, const auto& part_symb_b) {
                     auto get_storage_size = [](const auto& part_symb) {
                       auto* op = part_symb->operation();
                       return op->find_store(part_symb)->storage_size();
                     };
                     return get_storage_size(part_symb_a) > get_storage_size(part_symb_b);
                   });

  for (auto& part_symb : partition_symbols) {
    if (strategy->has_assignment(part_symb)) continue;

    auto* op       = part_symb->operation();
    auto store     = op->find_store(part_symb);
    auto partition = store->find_or_create_key_partition();
    if (!strategy->has_launch_domain(op)) {
      if (partition->has_launch_domain())
        strategy->set_launch_domain(op, partition->launch_domain());
      else
        strategy->set_single_launch(op);
    }

    std::vector<const Variable*> equiv_class;
    constraints.find_equivalence_class(part_symb, equiv_class);

    for (auto symb : equiv_class) strategy->insert(symb, partition);
  }

  return std::move(strategy);
}

}  // namespace legate
