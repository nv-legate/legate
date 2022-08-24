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

#include "core/runtime/operation.h"

#include <sstream>
#include <unordered_set>

#include "core/data/logical_store_detail.h"
#include "core/data/scalar.h"
#include "core/partitioning/constraint.h"
#include "core/partitioning/constraint_graph.h"
#include "core/partitioning/partition.h"
#include "core/partitioning/partitioner.h"
#include "core/runtime/context.h"
#include "core/runtime/launcher.h"
#include "core/runtime/req_analyzer.h"
#include "core/runtime/runtime.h"

namespace legate {

Operation::Operation(LibraryContext* library, uint64_t unique_id, int64_t mapper_id)
  : library_(library), unique_id_(unique_id), mapper_id_(mapper_id)
{
}

Operation::~Operation() {}

void Operation::add_input(LogicalStore store, const Variable* partition_symbol)
{
  add_store(inputs_, store, partition_symbol);
}

void Operation::add_output(LogicalStore store, const Variable* partition_symbol)
{
  add_store(outputs_, store, partition_symbol);
}

void Operation::add_reduction(LogicalStore store,
                              Legion::ReductionOpID redop,
                              const Variable* partition_symbol)
{
  add_store(reductions_, store, partition_symbol);
  reduction_ops_.push_back(redop);
}

void Operation::add_store(std::vector<StoreArg>& store_args,
                          LogicalStore& store,
                          const Variable* partition_symbol)
{
  auto store_impl = store.impl();
  store_args.push_back(StoreArg(store_impl.get(), partition_symbol));
  store_mappings_[*partition_symbol] = store_impl.get();
  all_stores_.insert(std::move(store_impl));
}

const Variable* Operation::declare_partition()
{
  partition_symbols_.emplace_back(new Variable(this, next_part_id_++));
  return partition_symbols_.back().get();
}

detail::LogicalStore* Operation::find_store(const Variable* part_symb) const
{
  auto finder = store_mappings_.find(*part_symb);
#ifdef DEBUG_LEGATE
  assert(store_mappings_.end() != finder);
#endif
  return finder->second;
}

void Operation::add_constraint(std::unique_ptr<Constraint> constraint)
{
  constraints_.push_back(std::move(constraint));
}

void Operation::add_to_constraint_graph(ConstraintGraph& constraint_graph) const
{
  for (auto& pair : inputs_) constraint_graph.add_partition_symbol(pair.second);
  for (auto& pair : outputs_) constraint_graph.add_partition_symbol(pair.second);
  for (auto& pair : reductions_) constraint_graph.add_partition_symbol(pair.second);
  for (auto& constraint : constraints_) constraint_graph.add_constraint(constraint.get());
}

Task::Task(LibraryContext* library, int64_t task_id, uint64_t unique_id, int64_t mapper_id /*=0*/)
  : Operation(library, unique_id, mapper_id), task_id_(task_id)
{
}

Task::~Task() {}

void Task::add_scalar_arg(const Scalar& scalar) { scalars_.push_back(scalar); }

struct field_size_fn {
  template <LegateTypeCode CODE>
  size_t operator()()
  {
    return sizeof(legate_type_of<CODE>);
  }
};

void Task::launch(Strategy* p_strategy)
{
  auto* runtime = Runtime::get_runtime();

  auto& strategy = *p_strategy;
  TaskLauncher launcher(library_, task_id_, mapper_id_);

  for (auto& pair : inputs_) {
    auto& store = pair.first;
    auto& var   = pair.second;
    auto proj   = strategy[var]->get_projection(store);
    launcher.add_input(store, std::move(proj));
  }
  for (auto& pair : outputs_) {
    auto& store = pair.first;
    if (store->unbound()) continue;
    auto& var = pair.second;
    auto part = strategy[var];
    auto proj = part->get_projection(store);
    launcher.add_output(store, std::move(proj));
    store->set_key_partition(part.get());
  }
  uint32_t idx = 0;
  for (auto& pair : reductions_) {
    auto& store = pair.first;
    auto& var   = pair.second;
    auto proj   = strategy[var]->get_projection(store);
    auto redop  = reduction_ops_[idx++];
    proj->set_reduction_op(redop);
    launcher.add_reduction(store, std::move(proj));
  }
  for (auto& pair : outputs_) {
    auto& store = pair.first;
    if (!store->unbound()) continue;
    auto& var        = pair.second;
    auto field_space = strategy.find_field_space(var);
    // TODO: We should reuse field ids here
    auto field_size = type_dispatch(store->code(), field_size_fn{});
    auto field_id   = runtime->allocate_field(field_space, field_size);
    launcher.add_unbound_output(store, field_space, field_id);
  }
  for (auto& scalar : scalars_) launcher.add_scalar(scalar);

  if (strategy.parallel(this))
    launcher.execute(strategy.launch_domain(this));
  else
    launcher.execute_single();
}

std::string Task::to_string() const
{
  std::stringstream ss;
  ss << library_->get_task_name(task_id_) << ":" << unique_id_;
  return ss.str();
}

}  // namespace legate
