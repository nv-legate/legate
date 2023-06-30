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

#include "core/operation/detail/task.h"

#include <sstream>

#include "core/data/scalar.h"
#include "core/operation/detail/projection.h"
#include "core/operation/detail/task_launcher.h"
#include "core/partitioning/constraint_solver.h"
#include "core/partitioning/partition.h"
#include "core/partitioning/partitioner.h"
#include "core/runtime/context.h"
#include "core/runtime/detail/communicator_manager.h"
#include "core/runtime/detail/provenance_manager.h"
#include "core/runtime/detail/runtime.h"

namespace legate::detail {

////////////////////////////////////////////////////
// legate::Task
////////////////////////////////////////////////////

Task::Task(const LibraryContext* library,
           int64_t task_id,
           uint64_t unique_id,
           mapping::MachineDesc&& machine)
  : Operation(unique_id, std::move(machine)), library_(library), task_id_(task_id)
{
}

void Task::add_scalar_arg(const Scalar& scalar) { scalars_.push_back(scalar); }

void Task::add_scalar_arg(Scalar&& scalar) { scalars_.emplace_back(std::move(scalar)); }

void Task::set_concurrent(bool concurrent) { concurrent_ = concurrent; }

void Task::set_side_effect(bool has_side_effect) { has_side_effect_ = has_side_effect; }

void Task::throws_exception(bool can_throw_exception)
{
  can_throw_exception_ = can_throw_exception;
}

void Task::add_communicator(const std::string& name)
{
  auto* comm_mgr = detail::Runtime::get_runtime()->communicator_manager();
  communicator_factories_.push_back(comm_mgr->find_factory(name));
}

void Task::launch(Strategy* p_strategy)
{
  auto& strategy = *p_strategy;
  detail::TaskLauncher launcher(library_, machine_, provenance_, task_id_);
  const auto* launch_domain = strategy.launch_domain(this);

  auto create_projection_info = [&strategy, &launch_domain](auto& store, auto& var) {
    auto store_partition = store->create_partition(strategy[var]);
    auto proj_info       = store_partition->create_projection_info(launch_domain);
    proj_info->tag       = strategy.is_key_partition(var) ? LEGATE_CORE_KEY_STORE_TAG : 0;
    return std::move(proj_info);
  };

  // Add input stores
  for (auto& [store, var] : inputs_) launcher.add_input(store, create_projection_info(store, var));

  // Add normal output stores
  for (auto& [store, var] : outputs_) {
    if (store->unbound()) continue;
    launcher.add_output(store, create_projection_info(store, var));
    store->set_key_partition(machine(), strategy[var].get());
  }

  // Add reduction stores
  uint32_t idx = 0;
  for (auto& [store, var] : reductions_) {
    auto store_partition = store->create_partition(strategy[var]);
    auto proj            = store_partition->create_projection_info(launch_domain);
    bool read_write      = store_partition->is_disjoint_for(launch_domain);
    auto redop           = reduction_ops_[idx++];
    proj->set_reduction_op(redop);
    launcher.add_reduction(store, std::move(proj), read_write);
  }

  // Add unbound output stores
  auto* runtime = detail::Runtime::get_runtime();
  for (auto& [store, var] : outputs_) {
    if (!store->unbound()) continue;
    auto field_space = strategy.find_field_space(var);
    // TODO: We should reuse field ids here
    auto field_size = store->type().size();
    auto field_id   = runtime->allocate_field(field_space, field_size);
    launcher.add_unbound_output(store, field_space, field_id);
  }

  // Add by-value scalars
  for (auto& scalar : scalars_) launcher.add_scalar(scalar);

  // Add communicators
  if (launch_domain != nullptr)
    for (auto* factory : communicator_factories_) {
      auto target = machine_.preferred_target;
      if (!factory->is_supported_target(target)) continue;
      auto& processor_range = machine_.processor_range();
      auto communicator     = factory->find_or_create(target, processor_range, *launch_domain);
      launcher.add_communicator(communicator);
      if (factory->needs_barrier()) launcher.set_insert_barrier(true);
    }

  launcher.set_side_effect(has_side_effect_);
  launcher.set_concurrent(concurrent_);
  launcher.throws_exception(can_throw_exception_);

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
  auto num_scalar_outs  = scalar_outputs_.size();
  auto num_scalar_reds  = scalar_reductions_.size();
  auto num_unbound_outs = unbound_outputs_.size();

  auto total = num_scalar_outs + num_scalar_reds + num_unbound_outs +
               static_cast<size_t>(can_throw_exception_);
  if (0 == total)
    return;
  else if (1 == total) {
    if (1 == num_scalar_outs) {
      auto [store, _] = outputs_[scalar_outputs_.front()];
      store->set_future(result);
    } else if (1 == num_scalar_reds) {
      auto [store, _] = reductions_[scalar_reductions_.front()];
      store->set_future(result);
    } else if (can_throw_exception_) {
      auto* runtime = detail::Runtime::get_runtime();
      runtime->record_pending_exception(result);
    }
#ifdef DEBUG_LEGATE
    else {
      assert(1 == num_unbound_outs);
    }
#endif
  } else {
    auto* runtime = detail::Runtime::get_runtime();
    uint32_t idx  = num_unbound_outs;
    for (const auto& out_idx : scalar_outputs_) {
      auto [store, _] = outputs_[out_idx];
      store->set_future(runtime->extract_scalar(result, idx++));
    }
    for (const auto& red_idx : scalar_reductions_) {
      auto [store, _] = reductions_[red_idx];
      store->set_future(runtime->extract_scalar(result, idx++));
    }
    if (can_throw_exception_)
      runtime->record_pending_exception(runtime->extract_scalar(result, idx));
  }
}

void Task::demux_scalar_stores(const Legion::FutureMap& result, const Domain& launch_domain)
{
  // Tasks with scalar outputs shouldn't have been parallelized
  assert(scalar_outputs_.empty());

  auto num_scalar_reds  = scalar_reductions_.size();
  auto num_unbound_outs = unbound_outputs_.size();

  auto total = num_scalar_reds + num_unbound_outs + static_cast<size_t>(can_throw_exception_);
  if (0 == total) return;

  auto* runtime = detail::Runtime::get_runtime();
  if (1 == total) {
    if (1 == num_scalar_reds) {
      auto red_idx    = scalar_reductions_.front();
      auto [store, _] = reductions_[red_idx];
      store->set_future(runtime->reduce_future_map(result, reduction_ops_[red_idx]));
    } else if (can_throw_exception_) {
      auto* runtime = detail::Runtime::get_runtime();
      runtime->record_pending_exception(runtime->reduce_exception_future_map(result));
    }
#ifdef DEBUG_LEGATE
    else {
      assert(1 == num_unbound_outs);
    }
#endif
  } else {
    uint32_t idx = num_unbound_outs;
    for (const auto& red_idx : scalar_reductions_) {
      auto [store, _] = reductions_[red_idx];
      auto values     = runtime->extract_scalar(result, idx++, launch_domain);
      store->set_future(runtime->reduce_future_map(values, reduction_ops_[red_idx]));
    }
    if (can_throw_exception_) {
      auto exn_fm = runtime->extract_scalar(result, idx, launch_domain);
      runtime->record_pending_exception(runtime->reduce_exception_future_map(exn_fm));
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

AutoTask::AutoTask(const LibraryContext* library,
                   int64_t task_id,
                   uint64_t unique_id,
                   mapping::MachineDesc&& machine)
  : Task(library, task_id, unique_id, std::move(machine))
{
}

void AutoTask::add_input(std::shared_ptr<LogicalStore> store, const Variable* partition_symbol)
{
  add_store(inputs_, std::move(store), partition_symbol);
}

void AutoTask::add_output(std::shared_ptr<LogicalStore> store, const Variable* partition_symbol)
{
  if (store->has_scalar_storage())
    scalar_outputs_.push_back(outputs_.size());
  else if (store->unbound())
    unbound_outputs_.push_back(outputs_.size());
  add_store(outputs_, std::move(store), partition_symbol);
}

void AutoTask::add_reduction(std::shared_ptr<LogicalStore> store,
                             Legion::ReductionOpID redop,
                             const Variable* partition_symbol)
{
  if (store->has_scalar_storage()) scalar_reductions_.push_back(reductions_.size());
  add_store(reductions_, std::move(store), partition_symbol);
  reduction_ops_.push_back(redop);
}

void AutoTask::add_store(std::vector<StoreArg>& store_args,
                         std::shared_ptr<LogicalStore> store,
                         const Variable* partition_symbol)
{
  store_args.push_back(StoreArg{store.get(), partition_symbol});
  record_partition(partition_symbol, std::move(store));
}

void AutoTask::add_constraint(std::unique_ptr<Constraint> constraint)
{
  constraints_.push_back(std::move(constraint));
}

void AutoTask::add_to_solver(detail::ConstraintSolver& solver)
{
  for (auto& constraint : constraints_) solver.add_constraint(constraint.get());
  for (auto& [_, symb] : outputs_) solver.add_partition_symbol(symb, true);
  for (auto& [_, symb] : reductions_) solver.add_partition_symbol(symb);
  for (auto& [_, symb] : inputs_) solver.add_partition_symbol(symb);
}

void AutoTask::validate()
{
  for (auto& constraint : constraints_) constraint->validate();
}

////////////////////////////////////////////////////
// legate::ManualTask
////////////////////////////////////////////////////

ManualTask::~ManualTask() {}

ManualTask::ManualTask(const LibraryContext* library,
                       int64_t task_id,
                       const Shape& launch_shape,
                       uint64_t unique_id,
                       mapping::MachineDesc&& machine)
  : Task(library, task_id, unique_id, std::move(machine)),
    strategy_(std::make_unique<detail::Strategy>())
{
  strategy_->set_launch_shape(this, launch_shape);
}

void ManualTask::add_input(std::shared_ptr<LogicalStore> store)
{
  add_store(inputs_, std::move(store), create_no_partition());
}

void ManualTask::add_input(std::shared_ptr<LogicalStorePartition> store_partition)
{
  add_store(inputs_, store_partition->store(), store_partition->partition());
}

void ManualTask::add_output(std::shared_ptr<LogicalStore> store)
{
  if (store->has_scalar_storage())
    scalar_outputs_.push_back(outputs_.size());
  else if (store->unbound())
    unbound_outputs_.push_back(outputs_.size());
  add_store(outputs_, std::move(store), create_no_partition());
}

void ManualTask::add_output(std::shared_ptr<LogicalStorePartition> store_partition)
{
#ifdef DEBUG_LEGATE
  // TODO: We need to raise an exception for the user error in this case
  assert(!store_partition->store()->unbound());
#endif
  if (store_partition->store()->has_scalar_storage()) scalar_outputs_.push_back(outputs_.size());
  add_store(outputs_, store_partition->store(), store_partition->partition());
}

void ManualTask::add_reduction(std::shared_ptr<LogicalStore> store, Legion::ReductionOpID redop)
{
  if (store->has_scalar_storage()) scalar_reductions_.push_back(reductions_.size());
  add_store(reductions_, std::move(store), create_no_partition());
  reduction_ops_.push_back(redop);
}

void ManualTask::add_reduction(std::shared_ptr<LogicalStorePartition> store_partition,
                               Legion::ReductionOpID redop)
{
  if (store_partition->store()->has_scalar_storage())
    scalar_reductions_.push_back(reductions_.size());
  add_store(reductions_, store_partition->store(), store_partition->partition());
  reduction_ops_.push_back(redop);
}

void ManualTask::add_store(std::vector<StoreArg>& store_args,
                           std::shared_ptr<LogicalStore> store,
                           std::shared_ptr<Partition> partition)
{
  auto partition_symbol = declare_partition();
  store_args.push_back(StoreArg{store.get(), partition_symbol});
  if (store->unbound()) {
    auto field_space = detail::Runtime::get_runtime()->create_field_space();
    strategy_->insert(partition_symbol, std::move(partition), field_space);
  } else
    strategy_->insert(partition_symbol, std::move(partition));
  all_stores_.insert(std::move(store));
}

void ManualTask::validate() {}

void ManualTask::launch(Strategy*) { Task::launch(strategy_.get()); }

void ManualTask::add_to_solver(ConstraintSolver& solver) {}

}  // namespace legate::detail
