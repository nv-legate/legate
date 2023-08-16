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

#include "core/operation/detail/task.h"

#include <sstream>

#include "core/data/scalar.h"
#include "core/operation/detail/launcher_arg.h"
#include "core/operation/detail/projection.h"
#include "core/operation/detail/task_launcher.h"
#include "core/partitioning/detail/constraint_solver.h"
#include "core/partitioning/detail/partitioner.h"
#include "core/partitioning/partition.h"
#include "core/runtime/detail/communicator_manager.h"
#include "core/runtime/detail/library.h"
#include "core/runtime/detail/provenance_manager.h"
#include "core/runtime/detail/runtime.h"
#include "core/type/detail/type_info.h"

namespace legate::detail {

////////////////////////////////////////////////////
// legate::Task
////////////////////////////////////////////////////

Task::Task(const Library* library,
           int64_t task_id,
           uint64_t unique_id,
           mapping::detail::Machine&& machine)
  : Operation(unique_id, std::move(machine)), library_(library), task_id_(task_id)
{
}

void Task::add_scalar_arg(const Scalar& scalar) { scalars_.push_back(scalar); }

void Task::add_scalar_arg(Scalar&& scalar) { scalars_.push_back(std::move(scalar)); }

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

void Task::record_scalar_output(std::shared_ptr<LogicalStore> store)
{
  scalar_outputs_.push_back(std::move(store));
}

void Task::record_unbound_output(std::shared_ptr<LogicalStore> store)
{
  unbound_outputs_.push_back(std::move(store));
}

void Task::record_scalar_reduction(std::shared_ptr<LogicalStore> store,
                                   Legion::ReductionOpID legion_redop_id)
{
  scalar_reductions_.push_back(std::make_pair(std::move(store), legion_redop_id));
}

void Task::launch_task(Strategy* p_strategy)
{
  auto& strategy = *p_strategy;
  detail::TaskLauncher launcher(library_, machine_, provenance_, task_id_);
  const auto* launch_domain = strategy.launch_domain(this);

  for (auto& [arr, mapping] : inputs_) {
    launcher.add_input(arr->to_launcher_arg(mapping, strategy, launch_domain, READ_ONLY, -1));
    for (auto& [store, symb] : mapping) {
      // Key partitions for unbound stores will be populated by the launcher
      if (store->unbound()) continue;
      store->set_key_partition(machine_, strategy[symb].get());
    }
  }
  for (auto& [arr, mapping] : outputs_) {
    launcher.add_output(arr->to_launcher_arg(mapping, strategy, launch_domain, WRITE_ONLY, -1));
  }
  uint32_t idx = 0;
  for (auto& [arr, mapping] : reductions_) {
    launcher.add_reduction(
      arr->to_launcher_arg(mapping, strategy, launch_domain, REDUCE, reduction_ops_[idx++]));
  }

  // Add by-value scalars
  for (auto& scalar : scalars_) launcher.add_scalar(std::move(scalar));

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
      scalar_outputs_.front()->set_future(result);
    } else if (1 == num_scalar_reds) {
      auto& [store, _] = scalar_reductions_.front();
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
    for (auto& store : scalar_outputs_) {
      store->set_future(runtime->extract_scalar(result, idx++));
    }
    for (auto& [store, _] : scalar_reductions_) {
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
      auto& [store, redop] = scalar_reductions_.front();
      store->set_future(runtime->reduce_future_map(result, redop));
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
    for (auto& [store, redop] : scalar_reductions_) {
      auto values = runtime->extract_scalar(result, idx++, launch_domain);
      store->set_future(runtime->reduce_future_map(values, redop));
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
  ss << library_->get_task_name(task_id_) << ":" << unique_id_;
  return std::move(ss).str();
}

////////////////////////////////////////////////////
// legate::AutoTask
////////////////////////////////////////////////////

AutoTask::AutoTask(const Library* library,
                   int64_t task_id,
                   uint64_t unique_id,
                   mapping::detail::Machine&& machine)
  : Task(library, task_id, unique_id, std::move(machine))
{
}

const Variable* AutoTask::add_input(std::shared_ptr<LogicalArray> array)
{
  auto symb = find_or_declare_partition(array);
  add_input(std::move(array), symb);
  return symb;
}

const Variable* AutoTask::add_output(std::shared_ptr<LogicalArray> array)
{
  auto symb = find_or_declare_partition(array);
  add_output(std::move(array), symb);
  return symb;
}

const Variable* AutoTask::add_reduction(std::shared_ptr<LogicalArray> array, int32_t redop)
{
  auto symb = find_or_declare_partition(array);
  add_reduction(std::move(array), redop, symb);
  return symb;
}

void AutoTask::add_input(std::shared_ptr<LogicalArray> array, const Variable* partition_symbol)
{
  auto& arg = inputs_.emplace_back(std::move(array));
  arg.array->generate_constraints(this, arg.mapping, partition_symbol);
  for (auto& [store, symb] : arg.mapping) record_partition(symb, store);
}

void AutoTask::add_output(std::shared_ptr<LogicalArray> array, const Variable* partition_symbol)
{
  array->record_scalar_or_unbound_outputs(this);
  // TODO: We will later support structs with list/string fields
  if (array->kind() == ArrayKind::LIST && array->unbound()) {
    arrays_to_fixup_.push_back(array.get());
  }
  auto& arg = outputs_.emplace_back(std::move(array));
  arg.array->generate_constraints(this, arg.mapping, partition_symbol);
  for (auto& [store, symb] : arg.mapping) record_partition(symb, store);
}

void AutoTask::add_reduction(std::shared_ptr<LogicalArray> array,
                             int32_t redop,
                             const Variable* partition_symbol)
{
  if (array->type()->variable_size()) {
    throw std::invalid_argument("List/string arrays cannot be used for reduction");
  }
  auto legion_redop_id = array->type()->find_reduction_operator(redop);
  array->record_scalar_reductions(this, legion_redop_id);
  reduction_ops_.push_back(legion_redop_id);

  auto& arg = reductions_.emplace_back(std::move(array));
  arg.array->generate_constraints(this, arg.mapping, partition_symbol);
  for (auto& [store, symb] : arg.mapping) record_partition(symb, store);
}

const Variable* AutoTask::find_or_declare_partition(std::shared_ptr<LogicalArray> array)
{
  return Operation::find_or_declare_partition(array->primary_store());
}

void AutoTask::add_constraint(std::unique_ptr<Constraint> constraint)
{
  constraints_.push_back(std::move(constraint));
}

void AutoTask::add_to_solver(detail::ConstraintSolver& solver)
{
  for (auto& constraint : constraints_) solver.add_constraint(std::move(constraint));
  for (auto& [_, mapping] : outputs_) {
    for (auto& [_, symb] : mapping) solver.add_partition_symbol(symb, true);
  }
  for (auto& [_, mapping] : inputs_) {
    for (auto& [_, symb] : mapping) solver.add_partition_symbol(symb, false);
  }
  for (auto& [_, mapping] : reductions_) {
    for (auto& [_, symb] : mapping) solver.add_partition_symbol(symb, false);
  }
}

void AutoTask::validate()
{
  for (auto& constraint : constraints_) constraint->validate();
}

void AutoTask::launch(Strategy* p_strategy)
{
  launch_task(p_strategy);
  if (!arrays_to_fixup_.empty()) fixup_ranges(*p_strategy);
}

void AutoTask::fixup_ranges(Strategy& strategy)
{
  const auto* launch_domain = strategy.launch_domain(this);
  if (nullptr == launch_domain) return;

  auto* core_lib = detail::Runtime::get_runtime()->core_library();
  detail::TaskLauncher launcher(core_lib, machine_, provenance_, LEGATE_CORE_FIXUP_RANGES);
  for (auto* array : arrays_to_fixup_) {
    launcher.add_output(array->to_launcher_arg_for_fixup(launch_domain, NO_ACCESS));
  }
  launcher.execute(*launch_domain);
}

////////////////////////////////////////////////////
// legate::ManualTask
////////////////////////////////////////////////////

ManualTask::~ManualTask() {}

ManualTask::ManualTask(const Library* library,
                       int64_t task_id,
                       const Shape& launch_shape,
                       uint64_t unique_id,
                       mapping::detail::Machine&& machine)
  : Task(library, task_id, unique_id, std::move(machine)),
    strategy_(std::make_unique<detail::Strategy>())
{
  if (!launch_shape.empty()) { strategy_->set_launch_shape(this, launch_shape); }
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
  if (store->has_scalar_storage()) {
    record_scalar_output(store);
  } else if (store->unbound()) {
    record_unbound_output(store);
  }
  add_store(outputs_, std::move(store), create_no_partition());
}

void ManualTask::add_output(std::shared_ptr<LogicalStorePartition> store_partition)
{
#ifdef DEBUG_LEGATE
  // TODO: We need to raise an exception for the user error in this case
  assert(!store_partition->store()->unbound());
#endif
  if (store_partition->store()->has_scalar_storage()) {
    record_scalar_output(store_partition->store());
  }
  add_store(outputs_, store_partition->store(), store_partition->partition());
}

void ManualTask::add_reduction(std::shared_ptr<LogicalStore> store, int32_t redop)
{
  auto legion_redop_id = store->type()->find_reduction_operator(redop);
  if (store->has_scalar_storage()) { record_scalar_reduction(store, legion_redop_id); }
  add_store(reductions_, std::move(store), create_no_partition());
  reduction_ops_.push_back(legion_redop_id);
}

void ManualTask::add_reduction(std::shared_ptr<LogicalStorePartition> store_partition,
                               int32_t redop)
{
  auto legion_redop_id = store_partition->store()->type()->find_reduction_operator(redop);
  if (store_partition->store()->has_scalar_storage()) {
    record_scalar_reduction(store_partition->store(), legion_redop_id);
  }
  add_store(reductions_, store_partition->store(), store_partition->partition());
  reduction_ops_.push_back(legion_redop_id);
}

void ManualTask::add_store(std::vector<ArrayArg>& store_args,
                           std::shared_ptr<LogicalStore> store,
                           std::shared_ptr<Partition> partition)
{
  auto partition_symbol = declare_partition();
  auto& arg             = store_args.emplace_back(std::make_shared<BaseLogicalArray>(store));
  arg.mapping.insert({store, partition_symbol});
  if (store->unbound()) {
    auto field_space = detail::Runtime::get_runtime()->create_field_space();
    strategy_->insert(partition_symbol, std::move(partition), field_space);
  } else {
    strategy_->insert(partition_symbol, std::move(partition));
  }
}

void ManualTask::validate() {}

void ManualTask::launch(Strategy*) { launch(); }

void ManualTask::launch() { launch_task(strategy_.get()); }

void ManualTask::add_to_solver(ConstraintSolver& solver) {}

}  // namespace legate::detail
