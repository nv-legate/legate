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

////////////////////////////////////////////////////
// legate::Operation
////////////////////////////////////////////////////

Operation::Operation(LibraryContext* library, uint64_t unique_id)
  : library_(library), unique_id_(unique_id)
{
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

////////////////////////////////////////////////////
// legate::Task
////////////////////////////////////////////////////

Task::Task(LibraryContext* library, int64_t task_id, uint64_t unique_id)
  : Operation(library, unique_id), task_id_(task_id)
{
}

void Task::add_scalar_arg(const Scalar& scalar) { scalars_.push_back(scalar); }

void Task::launch(Strategy* p_strategy)
{
  auto& strategy = *p_strategy;
  TaskLauncher launcher(library_, task_id_);
  auto launch_domain = strategy.launch_domain(this);
  auto launch_ndim   = launch_domain != nullptr ? launch_domain->dim : 0;

  for (auto& pair : inputs_) {
    auto& store = pair.first;
    auto& var   = pair.second;
    auto proj   = strategy[var]->get_projection(store, launch_ndim);
    launcher.add_input(store, std::move(proj));
  }
  for (auto& pair : outputs_) {
    auto& store = pair.first;
    if (store->unbound()) continue;
    auto& var = pair.second;
    auto part = strategy[var];
    auto proj = part->get_projection(store, launch_ndim);
    launcher.add_output(store, std::move(proj));
    store->set_key_partition(part.get());
  }
  uint32_t idx = 0;
  for (auto& pair : reductions_) {
    auto& store = pair.first;
    auto& var   = pair.second;
    auto proj   = strategy[var]->get_projection(store, launch_ndim);
    auto redop  = reduction_ops_[idx++];
    proj->set_reduction_op(redop);
    launcher.add_reduction(store, std::move(proj));
  }

  auto* runtime = Runtime::get_runtime();
  for (auto& pair : outputs_) {
    auto& store = pair.first;
    if (!store->unbound()) continue;
    auto& var        = pair.second;
    auto field_space = strategy.find_field_space(var);
    // TODO: We should reuse field ids here
    // FIXME: Need to catch up the type system change
    auto field_size = 0;  // store->type().size();
    auto field_id   = runtime->allocate_field(field_space, field_size);
    launcher.add_unbound_output(store, field_space, field_id);
  }
  for (auto& scalar : scalars_) launcher.add_scalar(scalar);

  if (launch_domain != nullptr) {
    auto result = launcher.execute(*launch_domain);
    demux_scalar_stores(result, *launch_domain);
  } else {
    auto result = launcher.execute_single();
    demux_scalar_stores(result);
  }
}

void Task::demux_scalar_stores(const Legion::Future& result)
{
  // TODO: Handle unbound stores
  auto num_scalar_outs = scalar_outputs_.size();
  auto num_scalar_reds = scalar_reductions_.size();

  auto total = num_scalar_outs + num_scalar_reds;
  if (0 == total)
    return;
  else if (1 == total) {
    if (1 == num_scalar_outs) {
      auto [store, _] = outputs_[scalar_outputs_.front()];
      store->set_future(result);
    } else {
      assert(1 == num_scalar_reds);
      auto [store, _] = reductions_[scalar_reductions_.front()];
      store->set_future(result);
    }
  } else {
    auto* runtime = Runtime::get_runtime();
    uint32_t idx  = 0;
    for (const auto& out_idx : scalar_outputs_) {
      auto [store, _] = outputs_[out_idx];
      store->set_future(runtime->extract_scalar(result, idx++));
    }
    for (const auto& red_idx : scalar_reductions_) {
      auto [store, _] = reductions_[red_idx];
      store->set_future(runtime->extract_scalar(result, idx++));
    }
  }
}

void Task::demux_scalar_stores(const Legion::FutureMap& result, const Legion::Domain& launch_domain)
{
  // Tasks with scalar outputs shouldn't have been parallelized
  assert(scalar_outputs_.empty());

  // TODO: Handle unbound stores
  auto num_scalar_reds = scalar_reductions_.size();

  auto total = num_scalar_reds;
  if (0 == total) return;

  auto* runtime = Runtime::get_runtime();
  if (1 == total) {
    assert(1 == num_scalar_reds);
    auto red_idx    = scalar_reductions_.front();
    auto [store, _] = reductions_[red_idx];
    store->set_future(runtime->reduce_future_map(result, reduction_ops_[red_idx]));
  } else {
    uint32_t idx = 0;
    for (const auto& red_idx : scalar_reductions_) {
      auto [store, _] = reductions_[red_idx];
      auto values     = runtime->extract_scalar(result, idx++, launch_domain);
      store->set_future(runtime->reduce_future_map(values, reduction_ops_[red_idx]));
    }
  }
}

std::string Task::to_string() const
{
  std::stringstream ss;
  ss << library_->find_task(task_id_)->name() << ":" << unique_id_;
  return std::move(ss).str();
}

////////////////////////////////////////////////////
// legate::AutoTask
////////////////////////////////////////////////////

AutoTask::AutoTask(LibraryContext* library, int64_t task_id, uint64_t unique_id)
  : Task(library, task_id, unique_id)
{
}

void AutoTask::add_input(LogicalStore store, const Variable* partition_symbol)
{
  add_store(inputs_, store, partition_symbol);
}

void AutoTask::add_output(LogicalStore store, const Variable* partition_symbol)
{
  if (store.impl()->has_scalar_storage()) scalar_outputs_.push_back(outputs_.size());
  add_store(outputs_, store, partition_symbol);
}

void AutoTask::add_reduction(LogicalStore store,
                             Legion::ReductionOpID redop,
                             const Variable* partition_symbol)
{
  if (store.impl()->has_scalar_storage()) scalar_reductions_.push_back(reductions_.size());
  add_store(reductions_, store, partition_symbol);
  reduction_ops_.push_back(redop);
}

void AutoTask::add_store(std::vector<StoreArg>& store_args,
                         LogicalStore& store,
                         const Variable* partition_symbol)
{
  auto store_impl = store.impl();
  store_args.push_back(StoreArg(store_impl.get(), partition_symbol));
  store_mappings_[*partition_symbol] = store_impl.get();
  all_stores_.insert(std::move(store_impl));
}

void AutoTask::add_constraint(std::unique_ptr<Constraint> constraint)
{
  constraints_.push_back(std::move(constraint));
}

void AutoTask::add_to_constraint_graph(ConstraintGraph& constraint_graph) const
{
  for (auto& constraint : constraints_) constraint_graph.add_constraint(constraint.get());
  for (auto& [_, symb] : inputs_) constraint_graph.add_partition_symbol(symb);
  for (auto& [_, symb] : outputs_) constraint_graph.add_partition_symbol(symb);
  for (auto& [_, symb] : reductions_) constraint_graph.add_partition_symbol(symb);
}

////////////////////////////////////////////////////
// legate::ManualTask
////////////////////////////////////////////////////

ManualTask::~ManualTask() {}

ManualTask::ManualTask(LibraryContext* library,
                       int64_t task_id,
                       const Shape& launch_shape,
                       uint64_t unique_id)
  : Task(library, task_id, unique_id), strategy_(std::make_unique<Strategy>())
{
  strategy_->set_launch_shape(this, launch_shape);
}

void ManualTask::add_input(LogicalStore store) { add_store(inputs_, store, create_no_partition()); }

void ManualTask::add_output(LogicalStore store)
{
  if (store.impl()->has_scalar_storage()) scalar_outputs_.push_back(outputs_.size());
  add_store(outputs_, store, create_no_partition());
}

void ManualTask::add_reduction(LogicalStore store, Legion::ReductionOpID redop)
{
  if (store.impl()->has_scalar_storage()) scalar_reductions_.push_back(reductions_.size());
  add_store(reductions_, store, create_no_partition());
  reduction_ops_.push_back(redop);
}

void ManualTask::add_input(LogicalStorePartition store_partition)
{
  add_store(inputs_, store_partition.store(), store_partition.partition());
}

void ManualTask::add_output(LogicalStorePartition store_partition)
{
  if (store_partition.store().impl()->has_scalar_storage())
    scalar_outputs_.push_back(outputs_.size());
  add_store(outputs_, store_partition.store(), store_partition.partition());
}

void ManualTask::add_reduction(LogicalStorePartition store_partition, Legion::ReductionOpID redop)
{
  if (store_partition.store().impl()->has_scalar_storage())
    scalar_reductions_.push_back(reductions_.size());
  add_store(reductions_, store_partition.store(), store_partition.partition());
  reduction_ops_.push_back(redop);
}

void ManualTask::add_store(std::vector<StoreArg>& store_args,
                           const LogicalStore& store,
                           std::shared_ptr<Partition> partition)
{
  auto store_impl       = store.impl();
  auto partition_symbol = declare_partition();
  store_args.push_back(StoreArg(store_impl.get(), partition_symbol));
  strategy_->insert(partition_symbol, std::move(partition));
  all_stores_.insert(std::move(store_impl));
}

void ManualTask::launch(Strategy*) { Task::launch(strategy_.get()); }

void ManualTask::add_to_constraint_graph(ConstraintGraph& constraint_graph) const {}

}  // namespace legate
