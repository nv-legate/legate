/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/task.h>

#include <legate_defines.h>

#include <legate/data/detail/array_tasks.h>
#include <legate/mapping/detail/mapping.h>
#include <legate/operation/detail/launcher_arg.h>
#include <legate/operation/detail/task_launcher.h>
#include <legate/partitioning/detail/constraint_solver.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/runtime/detail/communicator_manager.h>
#include <legate/runtime/detail/library.h>
#include <legate/runtime/detail/region_manager.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/task/detail/inline_task_body.h>
#include <legate/task/detail/returned_exception.h>
#include <legate/task/detail/task_info.h>
#include <legate/task/detail/task_return.h>
#include <legate/task/detail/task_return_layout.h>
#include <legate/task/detail/variant_info.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/cpp_version.h>
#include <legate/utilities/detail/align.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/detail/zip.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <stdexcept>

namespace legate::detail {

////////////////////////////////////////////////////
// legate::Task
////////////////////////////////////////////////////

Task::Task(const Library& library,
           const VariantInfo& variant_info,
           LocalTaskID task_id,
           std::uint64_t unique_id,
           std::int32_t priority,
           mapping::detail::Machine machine,
           bool can_inline_launch)
  : Operation{unique_id, priority, std::move(machine)},
    library_{library},
    vinfo_{variant_info},
    task_id_{task_id},
    concurrent_{variant_info.options.concurrent},
    has_side_effect_{variant_info.options.has_side_effect},
    can_throw_exception_{variant_info.options.may_throw_exception},
    can_elide_device_ctx_sync_{variant_info.options.elide_device_ctx_sync},
    can_inline_launch_{can_inline_launch}
{
  if (const auto& communicators = variant_info.options.communicators; communicators.has_value()) {
    for (auto&& comm : *communicators) {
      if (comm.empty()) {
        LEGATE_CPP_VERSION_TODO(20, "No need to use empty comm as sentinel value anymore");
        break;
      }
      add_communicator(comm, /* bypass_signature_check */ true);
    }
  }

  if (const auto& signature = variant_info.signature; signature.has_value()) {
    constexpr auto nargs_upper_limit = [](const std::optional<TaskSignature::Nargs>& nargs) {
      return nargs.has_value() ? nargs->upper_limit() : 0;
    };

    const auto& sig = *signature;

    inputs_.reserve(nargs_upper_limit(sig->inputs()));
    outputs_.reserve(nargs_upper_limit(sig->outputs()));
    scalars_.reserve(nargs_upper_limit(sig->scalars()));

    const auto num_redops = nargs_upper_limit(sig->redops());

    reductions_.reserve(num_redops);
    reduction_ops_.reserve(num_redops);
  }
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

void Task::add_communicator(std::string_view name, bool bypass_signature_check)
{
  if (const auto& comms = variant_info_().options.communicators; comms.has_value()) {
    if (!bypass_signature_check) {
      throw TracedException<std::runtime_error>{
        fmt::format("Task {} has pre-declared communicator(s) ({}), cannot override it by adding "
                    "communicators (of any kind) at the task callsite. Either remove the "
                    "communicator from the task declaration, or remove the manual "
                    "add_communicator() calls from the task construction sequence.",
                    *this,
                    *comms)};
    }
  }

  const auto& comm_mgr = detail::Runtime::get_runtime().communicator_manager();
  auto&& factory       = comm_mgr.find_factory(name);

  if (factory.is_supported_target(machine().preferred_target())) {
    communicator_factories_.emplace_back(factory);
    set_concurrent(true);
  }
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
                                   GlobalRedopID legion_redop_id)
{
  scalar_reductions_.emplace_back(std::move(store), legion_redop_id);
}

void Task::validate()
{
  Operation::validate();
  if (const auto& signature = variant_info_().signature) {
    (*signature)->check_signature(*this);
  }
}

void Task::inline_launch_() const
{
  const auto processor_kind = Runtime::get_runtime().get_executing_processor().kind();
  const auto variant_code   = mapping::detail::to_variant_code(processor_kind);
  const auto body           = [&] {
    const auto variant = library().find_task(local_task_id())->find_variant(variant_code);

    LEGATE_ASSERT(variant.has_value());
    return variant->get().body;  // NOLINT(bugprone-unchecked-optional-access)
  }();

  inline_task_body(*this, variant_code, body);
}

void Task::legion_launch_(Strategy* strategy_ptr)
{
  auto& strategy       = *strategy_ptr;
  auto launcher        = detail::TaskLauncher{library(), machine(), provenance(), local_task_id()};
  auto&& launch_domain = strategy.launch_domain(this);
  const auto valid_launch  = launch_domain.is_valid();
  const auto launch_volume = launch_domain.get_volume();

  launcher.set_priority(priority());

  for (auto&& [arr, mapping, projection] : inputs_) {
    launcher.add_input(arr->to_launcher_arg(
      mapping, strategy, launch_domain, projection, LEGION_READ_ONLY, GlobalRedopID{-1}));
  }

  for (auto&& [arr, mapping, projection] : outputs_) {
    launcher.add_output(arr->to_launcher_arg(
      mapping, strategy, launch_domain, projection, LEGION_WRITE_ONLY, GlobalRedopID{-1}));
  }

  for (auto&& [redop, rest] : legate::detail::zip_equal(reduction_ops_, reductions_)) {
    auto&& [arr, mapping, projection] = rest;

    launcher.add_reduction(
      arr->to_launcher_arg(mapping, strategy, launch_domain, projection, LEGION_REDUCE, redop));
  }

  // Add by-value scalars
  for (auto&& scalar : scalars_) {
    launcher.add_scalar(std::move(scalar));
  }

  // Add communicators
  if (valid_launch && (launch_volume > 1)) {
    const auto target           = machine().preferred_target();
    const auto& processor_range = machine().processor_range();

    // Use explicit type here in order to get the reference_wrapper to coerce
    for (CommunicatorFactory& factory : communicator_factories_) {
      launcher.add_communicator(factory.find_or_create(target, processor_range, launch_domain));
      if (factory.needs_barrier()) {
        launcher.set_insert_barrier(true);
      }
    }
  }

  launcher.set_side_effect(has_side_effect_);
  launcher.set_concurrent(concurrent_);
  launcher.throws_exception(can_throw_exception());
  launcher.can_elide_device_ctx_sync(can_elide_device_ctx_sync());
  launcher.set_future_size(calculate_future_size_());

  // TODO(wonchanl): Once we implement a precise interference checker, this workaround can be
  // removed
  constexpr auto has_projection = [](const auto& args) {
    return std::any_of(
      args.begin(), args.end(), [](const auto& arg) { return arg.projection.has_value(); });
  };
  launcher.relax_interference_checks(
    valid_launch &&
    (has_projection(inputs_) || has_projection(outputs_) || has_projection(reductions_)));

  if (valid_launch) {
    auto result = launcher.execute(launch_domain);

    if (launch_volume > 1) {
      demux_scalar_stores_(result, launch_domain);
    } else {
      demux_scalar_stores_(result.get_future(launch_domain.lo()));
    }
  } else {
    auto result = launcher.execute_single();

    demux_scalar_stores_(result);
  }
}

void Task::launch_task_(Strategy* strategy)
{
  auto launch_fast = can_inline_launch_;

  if (launch_fast) {
    // TODO(jfaibussowit)
    // Inline launch not yet supported for unbound arrays. Not yet clear what to do to
    // materialize the buffers (as sizes aren't yet known).
    launch_fast = std::none_of(outputs_.begin(), outputs_.end(), [](const TaskArrayArg& array_arg) {
      return array_arg.array->unbound();
    });
  }

  if (launch_fast) {
    inline_launch_();
  } else {
    legion_launch_(strategy);
  }
}

void Task::demux_scalar_stores_(const Legion::Future& result)
{
  auto num_scalar_outs  = scalar_outputs_.size();
  auto num_scalar_reds  = scalar_reductions_.size();
  auto num_unbound_outs = unbound_outputs_.size();

  auto total = num_scalar_outs + num_scalar_reds + num_unbound_outs +
               static_cast<std::size_t>(can_throw_exception());
  if (0 == total) {
    return;
  }
  if (1 == total) {
    if (1 == num_scalar_outs) {
      scalar_outputs_.front()->set_future(result);
    } else if (1 == num_scalar_reds) {
      scalar_reductions_.front().first->set_future(result);
    } else if (can_throw_exception()) {
      detail::Runtime::get_runtime().record_pending_exception(result);
    } else {
      LEGATE_ASSERT(1 == num_unbound_outs);
    }
  } else {
    auto&& runtime     = detail::Runtime::get_runtime();
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
    if (can_throw_exception()) {
      runtime.record_pending_exception(runtime.extract_scalar(
        result, return_layout.total_size(), legate::detail::ReturnedException::max_size()));
    }
  }
}

void Task::demux_scalar_stores_(const Legion::FutureMap& result, const Domain& launch_domain)
{
  auto num_scalar_outs  = scalar_outputs_.size();
  auto num_scalar_reds  = scalar_reductions_.size();
  auto num_unbound_outs = unbound_outputs_.size();

  auto total = num_scalar_outs + num_scalar_reds + num_unbound_outs +
               static_cast<std::size_t>(can_throw_exception());
  if (0 == total) {
    return;
  }

  auto&& runtime = detail::Runtime::get_runtime();
  if (1 == total) {
    if (1 == num_scalar_outs) {
      scalar_outputs_.front()->set_future_map(result);
    } else if (1 == num_scalar_reds) {
      auto& [store, redop] = scalar_reductions_.front();

      store->set_future(runtime.reduce_future_map(result, redop, store->get_future()));
    } else if (can_throw_exception()) {
      runtime.record_pending_exception(runtime.reduce_exception_future_map(result));
    } else {
      LEGATE_ASSERT(1 == num_unbound_outs);
    }
  } else {
    auto return_layout = TaskReturnLayoutForUnpack{num_unbound_outs * sizeof(std::size_t)};

    auto extract_future_map = [&](auto&& future_map, auto&& store) {
      auto size   = store->type()->size();
      auto offset = return_layout.next(size, store->type()->alignment());
      return runtime.extract_scalar(future_map, offset, size, launch_domain);
    };

    const auto compute_offset = [&](auto&& store) {
      return return_layout.next(store->type()->size(), store->type()->alignment());
    };

    for (auto&& store : scalar_outputs_) {
      store->set_future_map(result, compute_offset(store));
    }
    for (auto&& [store, redop] : scalar_reductions_) {
      auto values = extract_future_map(result, store);

      store->set_future(runtime.reduce_future_map(values, redop, store->get_future()));
    }
    if (can_throw_exception()) {
      auto exn_fm = runtime.extract_scalar(result,
                                           return_layout.total_size(),
                                           legate::detail::ReturnedException::max_size(),
                                           launch_domain);

      runtime.record_pending_exception(runtime.reduce_exception_future_map(exn_fm));
    }
  }
}

std::size_t Task::calculate_future_size_() const
{
  // Here we calculate the total size of buffers created for scalar stores and unbound stores in the
  // following way: each output or reduction scalar store embeds a buffer to hold the update; each
  // unbound store gets a buffer that holds the number of elements, which will later be retrieved to
  // keep track of the store's key partition (of the legate::detail::Weighted type).
  auto&& all_array_args = {std::cref(inputs()), std::cref(outputs()), std::cref(reductions())};
  auto layout           = TaskReturnLayoutForUnpack{};
  for (auto&& array_args : all_array_args) {
    for (auto&& [arr, mapping, projection] : array_args.get()) {
      arr->calculate_pack_size(&layout);
    }
  }

  // Align the buffer size to the 16-byte boundary (see the comment in task_return.cc for the
  // detail)
  return round_up_to_multiple(layout.total_size(), TaskReturn::ALIGNMENT);
}

std::string Task::to_string(bool show_provenance) const
{
  auto result = fmt::format("{}:{}", library().get_task_name(local_task_id()), unique_id_);

  if (!provenance().empty() && show_provenance) {
    fmt::format_to(std::back_inserter(result), "[{}]", provenance());
  }
  return result;
}

bool Task::needs_flush() const
{
  constexpr auto needs_flush = [](auto&& array_arg) { return array_arg.needs_flush(); };
  return can_throw_exception() || std::any_of(inputs_.begin(), inputs_.end(), needs_flush) ||
         std::any_of(outputs_.begin(), outputs_.end(), needs_flush) ||
         std::any_of(reductions_.begin(), reductions_.end(), needs_flush);
}

////////////////////////////////////////////////////
// legate::AutoTask
////////////////////////////////////////////////////

AutoTask::AutoTask(const Library& library,
                   const VariantInfo& variant_info,
                   LocalTaskID task_id,
                   std::uint64_t unique_id,
                   std::int32_t priority,
                   mapping::detail::Machine machine)
  : Task{library,
         variant_info,
         task_id,
         unique_id,
         priority,
         std::move(machine),
         /* can_inline_launch */ Config::get_config().enable_inline_task_launch()}
{
}

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

const Variable* AutoTask::add_reduction(InternalSharedPtr<LogicalArray> array,
                                        std::int32_t redop_kind)
{
  auto symb = find_or_declare_partition(array);
  add_reduction(std::move(array), redop_kind, symb);
  return symb;
}

void AutoTask::add_input(InternalSharedPtr<LogicalArray> array, const Variable* partition_symbol)
{
  if (array->unbound()) {
    throw TracedException<std::invalid_argument>{"Unbound arrays cannot be used as input"};
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
                             std::int32_t redop_kind,
                             const Variable* partition_symbol)
{
  if (array->unbound()) {
    throw TracedException<std::invalid_argument>{"Unbound arrays cannot be used for reductions"};
  }

  if (array->type()->variable_size()) {
    throw TracedException<std::invalid_argument>{"List/string arrays cannot be used for reduction"};
  }
  auto legion_redop_id = array->type()->find_reduction_operator(redop_kind);

  array->record_scalar_reductions(this, legion_redop_id);
  reduction_ops_.push_back(legion_redop_id);

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

void AutoTask::add_constraint(InternalSharedPtr<Constraint> constraint, bool bypass_signature_check)
{
  const auto have_signature = [&] {
    if (const auto& sig = variant_info_().signature; sig.has_value()) {
      LEGATE_ASSERT(*sig);
      return (*sig)->constraints().has_value();
    }
    return false;
  };

  if (!bypass_signature_check && have_signature()) {
    throw TracedException<std::runtime_error>{fmt::format(
      "Task {} has pre-declared signature, cannot override it by adding constraints (of any "
      "kind) at the task callsite. Either remove the signature from the task declaration, or "
      "remove the manual add_constraint() calls from the task construction sequence.",
      *this)};
  }

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
  Task::validate();
  if (const auto& signature = variant_info_().signature) {
    (*signature)->apply_constraints(this);
  }
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
  auto&& launch_domain = strategy.launch_domain(this);
  if (!launch_domain.is_valid()) {
    return;
  }

  auto&& runtime  = Runtime::get_runtime();
  auto&& core_lib = runtime.core_library();
  auto launcher =
    detail::TaskLauncher{core_lib, machine(), provenance(), FixupRanges::TASK_CONFIG.task_id()};

  launcher.set_priority(priority());

  for (auto* array : arrays_to_fixup_) {
    // TODO(wonchanl): We should pass projection functors here once we start supporting
    // string/list legate arrays in ManualTasks
    launcher.add_output(array->to_launcher_arg_for_fixup(launch_domain, LEGION_NO_ACCESS));
  }

  launcher.execute(launch_domain);
}

////////////////////////////////////////////////////
// legate::ManualTask
////////////////////////////////////////////////////

ManualTask::ManualTask(const Library& library,
                       const VariantInfo& variant_info,
                       LocalTaskID task_id,
                       const Domain& launch_domain,
                       std::uint64_t unique_id,
                       std::int32_t priority,
                       mapping::detail::Machine machine)
  : Task{library,
         variant_info,
         task_id,
         unique_id,
         priority,
         std::move(machine),
         /* can_inline_launch */ false}
{
  strategy_.set_launch_domain(this, launch_domain);
}

void ManualTask::add_input(const InternalSharedPtr<LogicalStore>& store)
{
  if (store->unbound()) {
    throw TracedException<std::invalid_argument>{"Unbound stores cannot be used as input"};
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
                               std::int32_t redop_kind)
{
  if (store->unbound()) {
    throw TracedException<std::invalid_argument>{"Unbound stores cannot be used for reduction"};
  }

  auto legion_redop_id = store->type()->find_reduction_operator(redop_kind);
  if (store->has_scalar_storage()) {
    record_scalar_reduction(store, legion_redop_id);
  }
  add_store_(reductions_, store, create_no_partition());
  reduction_ops_.push_back(legion_redop_id);
}

void ManualTask::add_reduction(const InternalSharedPtr<LogicalStorePartition>& store_partition,
                               std::int32_t redop_kind,
                               std::optional<SymbolicPoint> projection)
{
  auto legion_redop_id = store_partition->store()->type()->find_reduction_operator(redop_kind);

  if (store_partition->store()->has_scalar_storage()) {
    record_scalar_reduction(store_partition->store(), legion_redop_id);
  }
  add_store_(
    reductions_, store_partition->store(), store_partition->partition(), std::move(projection));
  reduction_ops_.push_back(legion_redop_id);
}

void ManualTask::add_store_(std::vector<TaskArrayArg>& store_args,
                            const InternalSharedPtr<LogicalStore>& store,
                            InternalSharedPtr<Partition> partition,
                            std::optional<SymbolicPoint> projection)
{
  const auto* partition_symbol = declare_partition();
  auto& arg =
    store_args.emplace_back(make_internal_shared<BaseLogicalArray>(store), std::move(projection));
  const auto unbound = store->unbound();

  arg.mapping.insert({store, partition_symbol});
  if (unbound) {
    auto&& runtime   = detail::Runtime::get_runtime();
    auto field_space = runtime.create_field_space();
    const auto field_id =
      runtime.allocate_field(field_space, RegionManager::FIELD_ID_BASE, store->type()->size());

    strategy_.insert(partition_symbol, std::move(partition), std::move(field_space), field_id);
  } else {
    strategy_.insert(partition_symbol, std::move(partition));
  }
}

void ManualTask::launch() { launch_task_(&strategy_); }

}  // namespace legate::detail
