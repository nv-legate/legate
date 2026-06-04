/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/task.h>

#include <legate_defines.h>

#include <legate/data/detail/logical_store.h>
#include <legate/data/detail/physical_store.h>
#include <legate/mapping/detail/mapping.h>
#include <legate/partitioning/detail/constraint_solver.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/partitioning/detail/partition/no_partition.h>
#include <legate/partitioning/detail/partition/opaque.h>
#include <legate/partitioning/detail/partitioner.h>
#include <legate/runtime/detail/communicator_manager.h>
#include <legate/runtime/detail/library.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/runtime/runtime.h>
#include <legate/task/detail/inline_task_body.h>
#include <legate/task/detail/returned_exception.h>
#include <legate/task/detail/task_info.h>
#include <legate/task/detail/variant_info.h>
#include <legate/task/task_context.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/cpp_version.h>
#include <legate/utilities/detail/align.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/detail/type_traits.h>
#include <legate/utilities/detail/zip.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/machine.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <stdexcept>

namespace legate::detail {

////////////////////////////////////////////////////
// legate::TaskBase
////////////////////////////////////////////////////

TaskBase::TaskBase(const Library& library,
                   const VariantInfo& variant_info,
                   LocalTaskID task_id,
                   std::uint64_t unique_id,
                   std::int32_t priority,
                   mapping::detail::Machine machine)
  : Operation{unique_id, priority, std::move(machine)},
    library_{library},
    vinfo_{variant_info},
    task_id_{task_id},
    concurrent_{variant_info.options.concurrent},
    has_side_effect_{variant_info.options.has_side_effect},
    can_throw_exception_{variant_info.options.may_throw_exception},
    can_elide_device_ctx_sync_{variant_info.options.elide_device_ctx_sync}
{
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

void TaskBase::add_scalar_arg(InternalSharedPtr<Scalar> scalar)
{
  scalars_.emplace_back(std::move(scalar));
}

void TaskBase::set_concurrent(bool concurrent) { concurrent_ = concurrent; }

void TaskBase::set_side_effect(bool has_side_effect) { has_side_effect_ = has_side_effect; }

void TaskBase::throws_exception(bool can_throw_exception)
{
  can_throw_exception_ = can_throw_exception;
}

void TaskBase::validate()
{
  Operation::validate();
  if (const auto& signature = variant_info_().signature) {
    (*signature)->check_signature(*this);
  }
}

std::string TaskBase::to_string(bool show_provenance) const
{
  auto result = fmt::format("{}:{}", library().get_task_name(local_task_id()), unique_id_);

  if (!provenance().empty() && show_provenance) {
    fmt::format_to(std::back_inserter(result), "[{}]", provenance());
  }
  return result;
}

bool TaskBase::needs_flush() const
{
  constexpr auto needs_flush = [](auto&& store_arg) { return store_arg.needs_flush(); };
  return can_throw_exception() || std::any_of(inputs_.begin(), inputs_.end(), needs_flush) ||
         std::any_of(outputs_.begin(), outputs_.end(), needs_flush) ||
         std::any_of(reductions_.begin(), reductions_.end(), needs_flush);
}

////////////////////////////////////////////////////
// legate::LogicalTask
////////////////////////////////////////////////////

LogicalTask::LogicalTask(const Library& library,
                         const VariantInfo& variant_info,
                         LocalTaskID task_id,
                         std::uint64_t unique_id,
                         std::int32_t priority,
                         mapping::detail::Machine machine)
  : TaskBase{library, variant_info, task_id, unique_id, priority, std::move(machine)}
{
  if (const auto& communicators = variant_info.options.communicators; communicators.has_value()) {
    for (auto&& comm : *communicators) {
      if (comm.empty()) {
        LEGATE_CPP_VERSION_TODO(20, "No need to use empty comm as sentinel value anymore");
        break;
      }
      // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
      add_communicator(comm, /* bypass_signature_check */ true);
    }
  }
}

void LogicalTask::set_streaming_generation(std::optional<StreamingGeneration> streaming_generation)
{
  streaming_gen_ = std::move(streaming_generation);
}

void LogicalTask::add_communicator(std::string_view name, bool bypass_signature_check)
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

void LogicalTask::record_scalar_output(InternalSharedPtr<LogicalStore> store)
{
  scalar_outputs_.push_back(std::move(store));
}

void LogicalTask::record_unbound_output(InternalSharedPtr<LogicalStore> store)
{
  store->get_storage()->set_bound(true /* bound */);
  unbound_outputs_.push_back(std::move(store));
}

void LogicalTask::record_scalar_reduction(InternalSharedPtr<LogicalStore> store,
                                          GlobalRedopID legion_redop_id)
{
  scalar_reductions_.emplace_back(std::move(store), legion_redop_id);
}

void LogicalTask::launch_task_(Strategy* strategy)
{
  inline_task_body(*this, machine().preferred_variant(), variant_info_().body, *strategy);
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
  : LogicalTask{library, variant_info, task_id, unique_id, priority, std::move(machine)}
{
}

const Variable* AutoTask::add_input(InternalSharedPtr<LogicalStore> store)
{
  const auto* symb = find_or_declare_partition(store);
  add_input(std::move(store), symb);
  return symb;
}

const Variable* AutoTask::add_output(InternalSharedPtr<LogicalStore> store)
{
  const auto* symb = find_or_declare_partition(store);
  add_output(std::move(store), symb);
  return symb;
}

const Variable* AutoTask::add_reduction(InternalSharedPtr<LogicalStore> store,
                                        std::int32_t redop_kind)
{
  const auto* symb = find_or_declare_partition(store);
  add_reduction(std::move(store), redop_kind, symb);
  return symb;
}

void AutoTask::add_input(InternalSharedPtr<LogicalStore> store, const Variable* partition_symbol)
{
  if (store->unbound()) {
    throw TracedException<std::invalid_argument>{"Unbound arrays cannot be used as input"};
  }

  auto& arg = inputs_.emplace_back(
    Legion::PrivilegeMode::LEGION_READ_ONLY, std::move(store), partition_symbol);

  record_partition_(partition_symbol, arg.store, AccessMode::READ);
}

void AutoTask::add_output(InternalSharedPtr<LogicalStore> store, const Variable* partition_symbol)
{
  if (store->has_scalar_storage()) {
    record_scalar_output(store);
  }
  if (store->unbound()) {
    record_unbound_output(store);
  }

  auto& arg = outputs_.emplace_back(
    Legion::PrivilegeMode::LEGION_WRITE_ONLY, std::move(store), partition_symbol);

  record_partition_(partition_symbol, arg.store, AccessMode::WRITE);
}

void AutoTask::add_reduction(InternalSharedPtr<LogicalStore> store,
                             std::int32_t redop_kind,
                             const Variable* partition_symbol)
{
  if (store->unbound()) {
    throw TracedException<std::invalid_argument>{"Unbound store cannot be used for reductions"};
  }

  LEGATE_ASSERT(!store->type()->variable_size());

  auto legion_redop_id = store->type()->find_reduction_operator(redop_kind);

  if (store->has_scalar_storage()) {
    record_scalar_reduction(store, legion_redop_id);
  }
  reduction_ops_.push_back(legion_redop_id);

  auto& arg = reductions_.emplace_back(
    Legion::PrivilegeMode::LEGION_REDUCE, std::move(store), partition_symbol);

  record_partition_(partition_symbol, arg.store, AccessMode::REDUCE);
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
    // TODO(amberhassaan): do not move constraints until
    // https://github.com/nv-legate/legate.internal/issues/3120 is resolved
    // solver.add_constraint(std::move(constraint));
    solver.add_constraint(constraint);
  }
  for (auto&& output : outputs_) {
    solver.add_partition_symbol(output.variable, AccessMode::WRITE);
    if (output.store->has_scalar_storage()) {
      solver.add_constraint(broadcast(output.variable));
    }
  }
  for (auto&& input : inputs_) {
    solver.add_partition_symbol(input.variable, AccessMode::READ);
  }
  for (auto&& reduction : reductions_) {
    solver.add_partition_symbol(reduction.variable, AccessMode::REDUCE);
  }
}

void AutoTask::validate()
{
  LogicalTask::validate();
  if (const auto& signature = variant_info_().signature) {
    (*signature)->apply_constraints(this);
  }
  for (auto&& constraint : constraints_) {
    constraint->validate();
  }
}

void AutoTask::launch(Strategy* p_strategy) { launch_task_(p_strategy); }

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
  : LogicalTask{library, variant_info, task_id, unique_id, priority, std::move(machine)},
    strategy_{make_internal_shared<Strategy>(this)}
{
  if (Runtime::get_runtime().config().enable_inline_task_launch() &&
      launch_domain.get_volume() > 1) {
    LEGATE_ABORT(
      fmt::format("ManualTask with inline task launch requires a single-point launch domain. "
                  "Instead, got domain with volume {}",
                  launch_domain.get_volume()));
  }

  strategy_->set_launch_domain(launch_domain);
}

const Domain& ManualTask::launch_domain() const { return strategy_->launch_domain(); }

void ManualTask::add_input(InternalSharedPtr<LogicalStore> store)
{
  if (store->unbound()) {
    throw TracedException<std::invalid_argument>{"Unbound stores cannot be used as input"};
  }

  add_store_(Legion::PrivilegeMode::LEGION_READ_ONLY, inputs_, store, create_no_partition());
}

void ManualTask::add_output(InternalSharedPtr<LogicalStore> store)
{
  add_store_(Legion::PrivilegeMode::LEGION_WRITE_ONLY, outputs_, store, create_no_partition());

  if (store->has_scalar_storage()) {
    record_scalar_output(store);
  }
  // It's important that we toggle the store's state to deferred bound only after we finish adding
  // it to the task. Otherwise, it's ambiguous whether the store was made deferred bound by this
  // task or an earlier task in the scheduling window.
  if (store->unbound()) {
    record_unbound_output(store);
  }
}

void ManualTask::add_reduction(InternalSharedPtr<LogicalStore> store, std::int32_t redop_kind)
{
  if (store->unbound()) {
    throw TracedException<std::invalid_argument>{"Unbound stores cannot be used for reduction"};
  }

  add_store_(Legion::PrivilegeMode::LEGION_REDUCE, reductions_, store, create_no_partition());

  auto legion_redop_id = store->type()->find_reduction_operator(redop_kind);

  reduction_ops_.push_back(legion_redop_id);
  if (store->has_scalar_storage()) {
    record_scalar_reduction(store, legion_redop_id);
  }
}

void ManualTask::add_input(const InternalSharedPtr<LogicalStorePartition>& store_partition,
                           std::optional<SymbolicPoint> projection,
                           bool is_key_partition)
{
  add_store_(Legion::PrivilegeMode::LEGION_READ_ONLY,
             inputs_,
             store_partition->store(),
             store_partition->partition(),
             std::move(projection),
             is_key_partition);
}

void ManualTask::add_output(const InternalSharedPtr<LogicalStorePartition>& store_partition,
                            std::optional<SymbolicPoint> projection,
                            bool is_key_partition)
{
  // Unbound stores can never be partitioned immediately
  LEGATE_ASSERT(!store_partition->store()->unbound());
  add_store_(Legion::PrivilegeMode::LEGION_WRITE_ONLY,
             outputs_,
             store_partition->store(),
             store_partition->partition(),
             std::move(projection),
             is_key_partition);
  if (store_partition->store()->has_scalar_storage()) {
    record_scalar_output(store_partition->store());
  }
}

void ManualTask::add_reduction(const InternalSharedPtr<LogicalStorePartition>& store_partition,
                               std::int32_t redop_kind,
                               std::optional<SymbolicPoint> projection,
                               bool is_key_partition)
{
  add_store_(Legion::PrivilegeMode::LEGION_REDUCE,
             reductions_,
             store_partition->store(),
             store_partition->partition(),
             std::move(projection),
             is_key_partition);

  auto legion_redop_id = store_partition->store()->type()->find_reduction_operator(redop_kind);

  if (store_partition->store()->has_scalar_storage()) {
    record_scalar_reduction(store_partition->store(), legion_redop_id);
  }
  reduction_ops_.push_back(legion_redop_id);
}

void ManualTask::add_store_(Legion::PrivilegeMode priv,
                            SmallVector<TaskStoreArg>& store_args,
                            const InternalSharedPtr<LogicalStore>& store,
                            InternalSharedPtr<Partition> partition,
                            std::optional<SymbolicPoint> projection,
                            bool is_key_partition)
{
  const auto* partition_symbol = declare_partition();

  store_args.emplace_back(priv, store, partition_symbol);

  const auto privelege_to_access_mode = [](Legion::PrivilegeMode p) {
    switch (p) {
      case LEGION_READ_ONLY: [[fallthrough]];
      case LEGION_READ_DISCARD: return AccessMode::READ;
      case LEGION_REDUCE: return AccessMode::REDUCE;
      case LEGION_WRITE_ONLY: [[fallthrough]];
      case LEGION_READ_WRITE: [[fallthrough]];
      case LEGION_WRITE_DISCARD: [[fallthrough]];
      case LEGION_WRITE_PRIV: return AccessMode::WRITE;
      case LEGION_NO_ACCESS: [[fallthrough]];
      case LEGION_DISCARD_MASK: [[fallthrough]];
      case LEGION_DISCARD_OUTPUT_MASK:
        LEGATE_ABORT(fmt::format("No way to convert privilege {} to AccessMode", p));
        return AccessMode::REDUCE;
    }
    LEGATE_ABORT(fmt::format("No way to convert privilege {} to AccessMode", p));
    return AccessMode::WRITE;
  };

  record_partition_(partition_symbol, store, privelege_to_access_mode(priv));

  if (is_key_partition) {
    strategy_->record_key_partition({}, *partition_symbol);
  }

  if (strategy_->parallel()) {
    const auto& transform = store->transform();

    if (projection.has_value()) {
      strategy_->insert_store_projection(
        Strategy::PrivateKey{}, *partition_symbol, transform->invert(projection.value()));
    } else if (partition->has_color_shape()) {
      strategy_->insert_store_projection(
        Strategy::PrivateKey{},
        *partition_symbol,
        Partitioner::infer_store_projection(
          strategy_->launch_domain(), partition->color_shape(), transform));
    }
  }

  if (!store->unbound()) {
    strategy_->insert(*partition_symbol, std::move(partition));
    return;
  }

  LEGATE_ABORT("Unbound stores are not supported yet");
}

void ManualTask::launch() { launch_task_(strategy_.get()); }

}  // namespace legate::detail
