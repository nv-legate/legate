/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "core/operation/detail/launcher_arg.h"
#include "core/operation/detail/task_launcher.h"
#include "core/partitioning/detail/constraint_solver.h"
#include "core/partitioning/partition.h"
#include "core/runtime/detail/communicator_manager.h"
#include "core/runtime/detail/library.h"
#include "core/runtime/detail/region_manager.h"
#include "core/runtime/detail/runtime.h"
#include "core/task/detail/task_return_layout.h"
#include "core/type/detail/type_info.h"
#include "core/utilities/detail/zip.h"

#include <fmt/format.h>

namespace legate::detail {

////////////////////////////////////////////////////
// legate::Task
////////////////////////////////////////////////////

Task::Task(const Library* library,
           std::int64_t task_id,
           std::uint64_t unique_id,
           std::int32_t priority,
           mapping::detail::Machine machine)
  : Operation{unique_id, priority, std::move(machine)}, library_{library}, task_id_{task_id}
{
}

void Task::add_scalar_arg(InternalSharedPtr<Scalar> scalar)
{
  scalars_.emplace_back(std::move(scalar));
}

void Task::set_concurrent(bool concurrent) { concurrent_ = concurrent; }

void Task::set_side_effect(bool has_side_effect) { has_side_effect_ = has_side_effect; }

void Task::throws_exception(bool can_throw_exception)
{
  can_throw_exception_ = can_throw_exception;
}

void Task::add_communicator(std::string_view name)
{
  auto* comm_mgr = detail::Runtime::get_runtime()->communicator_manager();
  communicator_factories_.push_back(comm_mgr->find_factory(std::move(name)));
}

void Task::record_scalar_output(InternalSharedPtr<LogicalStore> store)
{
  scalar_outputs_.push_back(std::move(store));
}

void Task::record_unbound_output(InternalSharedPtr<LogicalStore> store)
{
  unbound_outputs_.push_back(std::move(store));
}

void Task::record_scalar_reduction(InternalSharedPtr<LogicalStore> store,
                                   Legion::ReductionOpID legion_redop_id)
{
  scalar_reductions_.emplace_back(std::move(store), legion_redop_id);
}

void Task::launch_task_(Strategy* p_strategy)
{
  auto& strategy     = *p_strategy;
  auto launcher      = detail::TaskLauncher{library_, machine_, provenance(), task_id_};
  auto launch_domain = strategy.launch_domain(this);

  launcher.set_priority(priority());

  for (auto&& [arr, mapping, projection] : inputs_) {
    launcher.add_input(
      arr->to_launcher_arg(mapping, strategy, launch_domain, projection, LEGION_READ_ONLY, -1));
  }

  for (auto&& [arr, mapping, projection] : outputs_) {
    launcher.add_output(
      arr->to_launcher_arg(mapping, strategy, launch_domain, projection, LEGION_WRITE_ONLY, -1));
  }

  for (auto&& [redop, rest] : legate::detail::zip_equal(reduction_ops_, reductions_)) {
    auto&& [arr, mapping, projection] = rest;

    launcher.add_reduction(
      arr->to_launcher_arg(mapping, strategy, launch_domain, projection, LEGION_REDUCE, redop));
  }

  // Add by-value scalars
  for (auto&& scalar : scalars_) {
    // TODO(jfaibussowit)
    // Copy is deliberate, we do not want to move out of scalar, since that would invalidate
    // the user-held scalar. Rather, launcher.add_scalar() should accept a InternalSharedPtr
    // argument instead...
    auto scal = *scalar;

    launcher.add_scalar(std::move(scal));
  }

  // Add communicators
  if (launch_domain.is_valid() && launch_domain.get_volume() > 1) {
    for (auto* factory : communicator_factories_) {
      auto target = machine_.preferred_target;

      if (!factory->is_supported_target(target)) {
        continue;
      }
      auto& processor_range = machine_.processor_range();
      auto communicator     = factory->find_or_create(target, processor_range, launch_domain);

      launcher.add_communicator(communicator);
      if (factory->needs_barrier()) {
        launcher.set_insert_barrier(true);
      }
    }
  }

  launcher.set_side_effect(has_side_effect_);
  launcher.set_concurrent(concurrent_);
  launcher.throws_exception(can_throw_exception_);

  // TODO(wonchanl): Once we implement a precise interference checker, this workaround can be
  // removed
  auto has_projection = [](auto& args) {
    return std::any_of(
      args.begin(), args.end(), [](const auto& arg) { return arg.projection.has_value(); });
  };
  launcher.relax_interference_checks(
    launch_domain.is_valid() &&
    (has_projection(inputs_) || has_projection(outputs_) || has_projection(reductions_)));

  if (launch_domain.is_valid()) {
    auto result = launcher.execute(launch_domain);

    if (launch_domain.get_volume() > 1) {
      demux_scalar_stores_(result, launch_domain);
    } else {
      demux_scalar_stores_(result.get_future(launch_domain.lo()));
    }
  } else {
    auto result = launcher.execute_single();

    demux_scalar_stores_(result);
  }
}

void Task::demux_scalar_stores_(const Legion::Future& result)
{
  auto num_scalar_outs  = scalar_outputs_.size();
  auto num_scalar_reds  = scalar_reductions_.size();
  auto num_unbound_outs = unbound_outputs_.size();

  auto total = num_scalar_outs + num_scalar_reds + num_unbound_outs +
               static_cast<std::size_t>(can_throw_exception_);
  if (0 == total) {
    return;
  }
  if (1 == total) {
    if (1 == num_scalar_outs) {
      scalar_outputs_.front()->set_future(result);
    } else if (1 == num_scalar_reds) {
      scalar_reductions_.front().first->set_future(result);
    } else if (can_throw_exception_) {
      detail::Runtime::get_runtime()->record_pending_exception(result);
    } else {
      LEGATE_ASSERT(1 == num_unbound_outs);
    }
  } else {
    auto* runtime      = detail::Runtime::get_runtime();
    auto return_layout = TaskReturnLayoutForUnpack{num_unbound_outs * sizeof(std::size_t)};

    const auto compute_offset = [&](auto&& store) {
      return return_layout.next(store->type()->size(), store->type()->alignment());
    };

    for (auto&& store : scalar_outputs_) {
      store->set_future(result, compute_offset(store));
    }
    for (auto&& [store, _] : scalar_reductions_) {
      store->set_future(result, compute_offset(store));
    }
    if (can_throw_exception_) {
      runtime->record_pending_exception(runtime->extract_scalar(
        result, return_layout.total_size(), std::numeric_limits<std::uint32_t>::max()));
    }
  }
}

void Task::demux_scalar_stores_(const Legion::FutureMap& result, const Domain& launch_domain)
{
  auto num_scalar_outs  = scalar_outputs_.size();
  auto num_scalar_reds  = scalar_reductions_.size();
  auto num_unbound_outs = unbound_outputs_.size();

  auto total = num_scalar_outs + num_scalar_reds + num_unbound_outs +
               static_cast<std::size_t>(can_throw_exception_);
  if (0 == total) {
    return;
  }

  const auto runtime = detail::Runtime::get_runtime();
  if (1 == total) {
    if (1 == num_scalar_outs) {
      scalar_outputs_.front()->set_future_map(result);
    } else if (1 == num_scalar_reds) {
      auto& [store, redop] = scalar_reductions_.front();

      store->set_future(runtime->reduce_future_map(result, redop, store->get_future()));
    } else if (can_throw_exception_) {
      runtime->record_pending_exception(runtime->reduce_exception_future_map(result));
    } else {
      LEGATE_ASSERT(1 == num_unbound_outs);
    }
  } else {
    auto return_layout = TaskReturnLayoutForUnpack{num_unbound_outs * sizeof(std::size_t)};

    auto extract_future_map = [&](auto&& future_map, auto&& store) {
      auto size   = store->type()->size();
      auto offset = return_layout.next(size, store->type()->alignment());
      return runtime->extract_scalar(future_map, offset, size, launch_domain);
    };

    const auto compute_offset = [&](auto&& store) {
      return return_layout.next(store->type()->size(), store->type()->alignment());
    };

    for (auto&& store : scalar_outputs_) {
      store->set_future_map(result, compute_offset(store));
    }
    for (auto&& [store, redop] : scalar_reductions_) {
      auto values = extract_future_map(result, store);

      store->set_future(runtime->reduce_future_map(values, redop, store->get_future()));
    }
    if (can_throw_exception_) {
      auto exn_fm = runtime->extract_scalar(result,
                                            return_layout.total_size(),
                                            std::numeric_limits<std::uint32_t>::max(),
                                            launch_domain);

      runtime->record_pending_exception(runtime->reduce_exception_future_map(exn_fm));
    }
  }
}

std::string Task::to_string() const
{
  auto result = fmt::format("{}:{}", library_->get_task_name(task_id_), unique_id_);

  if (!provenance().empty()) {
    fmt::format_to(std::back_inserter(result), "[{}]", provenance());
  }
  return result;
}

////////////////////////////////////////////////////
// legate::AutoTask
////////////////////////////////////////////////////

const Variable* AutoTask::add_input(InternalSharedPtr<LogicalArray> array)
{
  auto symb = find_or_declare_partition(array);
  add_input(std::move(array), symb);
  return symb;
}

const Variable* AutoTask::add_output(InternalSharedPtr<LogicalArray> array)
{
  auto symb = find_or_declare_partition(array);
  add_output(std::move(array), symb);
  return symb;
}

const Variable* AutoTask::add_reduction(InternalSharedPtr<LogicalArray> array, std::int32_t redop)
{
  auto symb = find_or_declare_partition(array);
  add_reduction(std::move(array), redop, symb);
  return symb;
}

void AutoTask::add_input(InternalSharedPtr<LogicalArray> array, const Variable* partition_symbol)
{
  if (array->unbound()) {
    throw std::invalid_argument{"Unbound arrays cannot be used as input"};
  }

  auto& arg = inputs_.emplace_back(std::move(array));

  arg.array->generate_constraints(this, arg.mapping, partition_symbol);
  for (auto&& [store, symb] : arg.mapping) {
    record_partition_(symb, store);
  }
}

void AutoTask::add_output(InternalSharedPtr<LogicalArray> array, const Variable* partition_symbol)
{
  array->record_scalar_or_unbound_outputs(this);
  // TODO(wonchanl): We will later support structs with list/string fields
  if (array->kind() == ArrayKind::LIST && array->unbound()) {
    arrays_to_fixup_.push_back(array.get());
  }
  auto& arg = outputs_.emplace_back(std::move(array));

  arg.array->generate_constraints(this, arg.mapping, partition_symbol);
  for (auto&& [store, symb] : arg.mapping) {
    record_partition_(symb, store);
  }
}

void AutoTask::add_reduction(InternalSharedPtr<LogicalArray> array,
                             std::int32_t redop,
                             const Variable* partition_symbol)
{
  if (array->unbound()) {
    throw std::invalid_argument{"Unbound arrays cannot be used for reductions"};
  }

  if (array->type()->variable_size()) {
    throw std::invalid_argument{"List/string arrays cannot be used for reduction"};
  }
  auto legion_redop_id = array->type()->find_reduction_operator(redop);

  array->record_scalar_reductions(this, static_cast<Legion::ReductionOpID>(legion_redop_id));
  reduction_ops_.push_back(static_cast<Legion::ReductionOpID>(legion_redop_id));

  auto& arg = reductions_.emplace_back(std::move(array));

  arg.array->generate_constraints(this, arg.mapping, partition_symbol);
  for (auto&& [store, symb] : arg.mapping) {
    record_partition_(symb, store);
  }
}

const Variable* AutoTask::find_or_declare_partition(const InternalSharedPtr<LogicalArray>& array)
{
  return Operation::find_or_declare_partition(array->primary_store());
}

void AutoTask::add_constraint(InternalSharedPtr<Constraint> constraint)
{
  constraints_.push_back(std::move(constraint));
}

void AutoTask::add_to_solver(detail::ConstraintSolver& solver)
{
  for (auto&& constraint : constraints_) {
    solver.add_constraint(std::move(constraint));
  }
  for (auto&& output : outputs_) {
    for (auto&& [store, symb] : output.mapping) {
      solver.add_partition_symbol(symb, AccessMode::WRITE);
      if (store->has_scalar_storage()) {
        solver.add_constraint(broadcast(symb));
      }
    }
  }
  for (auto&& input : inputs_) {
    for (auto&& [_, symb] : input.mapping) {
      solver.add_partition_symbol(symb, AccessMode::READ);
    }
  }
  for (auto&& reduction : reductions_) {
    for (auto&& [_, symb] : reduction.mapping) {
      solver.add_partition_symbol(symb, AccessMode::REDUCE);
    }
  }
}

void AutoTask::validate()
{
  for (auto&& constraint : constraints_) {
    constraint->validate();
  }
}

void AutoTask::launch(Strategy* p_strategy)
{
  launch_task_(p_strategy);
  if (!arrays_to_fixup_.empty()) {
    fixup_ranges_(*p_strategy);
  }
}

void AutoTask::fixup_ranges_(Strategy& strategy)
{
  auto launch_domain = strategy.launch_domain(this);
  if (!launch_domain.is_valid()) {
    return;
  }

  auto* core_lib = detail::Runtime::get_runtime()->core_library();
  auto launcher  = detail::TaskLauncher{core_lib, machine_, provenance(), LEGATE_CORE_FIXUP_RANGES};

  launcher.set_priority(priority());

  for (auto* array : arrays_to_fixup_) {
    // TODO(wonchanl): We should pass projection functors here once we start supporting string/list
    // legate arrays in ManualTasks
    launcher.add_output(array->to_launcher_arg_for_fixup(launch_domain, NO_ACCESS));
  }
  launcher.execute(launch_domain);
}

////////////////////////////////////////////////////
// legate::ManualTask
////////////////////////////////////////////////////

ManualTask::ManualTask(const Library* library,
                       std::int64_t task_id,
                       const Domain& launch_domain,
                       std::uint64_t unique_id,
                       std::int32_t priority,
                       mapping::detail::Machine machine)
  : Task{library, task_id, unique_id, priority, std::move(machine)},
    strategy_{std::make_unique<detail::Strategy>()}
{
  strategy_->set_launch_domain(this, launch_domain);
}

void ManualTask::add_input(const InternalSharedPtr<LogicalStore>& store)
{
  if (store->unbound()) {
    throw std::invalid_argument{"Unbound stores cannot be used as input"};
  }

  add_store_(inputs_, store, create_no_partition());
}

void ManualTask::add_input(const InternalSharedPtr<LogicalStorePartition>& store_partition,
                           std::optional<SymbolicPoint> projection)
{
  add_store_(
    inputs_, store_partition->store(), store_partition->partition(), std::move(projection));
}

void ManualTask::add_output(const InternalSharedPtr<LogicalStore>& store)
{
  if (store->has_scalar_storage()) {
    record_scalar_output(store);
  } else if (store->unbound()) {
    record_unbound_output(store);
  }
  add_store_(outputs_, store, create_no_partition());
}

void ManualTask::add_output(const InternalSharedPtr<LogicalStorePartition>& store_partition,
                            std::optional<SymbolicPoint> projection)
{
  // TODO(wonchanl): We need to raise an exception for the user error in this case
  LEGATE_ASSERT(!store_partition->store()->unbound());
  if (store_partition->store()->has_scalar_storage()) {
    record_scalar_output(store_partition->store());
  }
  add_store_(
    outputs_, store_partition->store(), store_partition->partition(), std::move(projection));
}

void ManualTask::add_reduction(const InternalSharedPtr<LogicalStore>& store,
                               Legion::ReductionOpID redop)
{
  if (store->unbound()) {
    throw std::invalid_argument{"Unbound stores cannot be used for reduction"};
  }

  auto legion_redop_id = store->type()->find_reduction_operator(redop);
  if (store->has_scalar_storage()) {
    record_scalar_reduction(store, static_cast<Legion::ReductionOpID>(legion_redop_id));
  }
  add_store_(reductions_, store, create_no_partition());
  reduction_ops_.push_back(static_cast<Legion::ReductionOpID>(legion_redop_id));
}

void ManualTask::add_reduction(const InternalSharedPtr<LogicalStorePartition>& store_partition,
                               Legion::ReductionOpID redop,
                               std::optional<SymbolicPoint> projection)
{
  auto legion_redop_id =
    static_cast<std::int32_t>(store_partition->store()->type()->find_reduction_operator(redop));

  if (store_partition->store()->has_scalar_storage()) {
    record_scalar_reduction(store_partition->store(),
                            static_cast<Legion::ReductionOpID>(legion_redop_id));
  }
  add_store_(
    reductions_, store_partition->store(), store_partition->partition(), std::move(projection));
  reduction_ops_.push_back(static_cast<Legion::ReductionOpID>(legion_redop_id));
}

void ManualTask::add_store_(std::vector<ArrayArg>& store_args,
                            const InternalSharedPtr<LogicalStore>& store,
                            InternalSharedPtr<Partition> partition,
                            std::optional<SymbolicPoint> projection)
{
  auto partition_symbol = declare_partition();
  auto& arg =
    store_args.emplace_back(make_internal_shared<BaseLogicalArray>(store), std::move(projection));
  const auto unbound = store->unbound();

  arg.mapping.insert({store, partition_symbol});
  if (unbound) {
    auto* runtime    = detail::Runtime::get_runtime();
    auto field_space = runtime->create_field_space();
    auto field_id =
      runtime->allocate_field(field_space, RegionManager::FIELD_ID_BASE, store->type()->size());

    strategy_->insert(partition_symbol, std::move(partition), field_space, field_id);
  } else {
    strategy_->insert(partition_symbol, std::move(partition));
  }
}

}  // namespace legate::detail
