/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/task.h>

#include <legate_defines.h>

#include <legate/cuda/detail/cuda_driver_api.h>
#include <legate/data/detail/copy_request.h>
#include <legate/data/detail/data_movement_requests.h>
#include <legate/data/detail/logical_store.h>
#include <legate/data/detail/physical_store.h>
#include <legate/data/detail/reduction_request.h>
#include <legate/mapping/detail/mapping.h>
#include <legate/partitioning/detail/constraint_solver.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/partitioning/detail/partition/no_partition.h>
#include <legate/partitioning/detail/partitioner.h>
#include <legate/runtime/detail/communicator_manager.h>
#include <legate/runtime/detail/library.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/runtime/runtime.h>
#include <legate/task/detail/task.h>
#include <legate/task/detail/task_context.h>
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
#include <legate/utilities/typedefs.h>

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

namespace {

void show_progress(const Strategy& strategy,
                   std::string_view task_name,
                   VariantCode variant_code,
                   std::string_view provenance)
{
  if (!Runtime::get_runtime().config().show_progress_requested()) {
    return;
  }

  const auto variant_str = [&]() -> std::string_view {
    switch (variant_code) {
      case VariantCode::CPU: return "CPU";
      case VariantCode::GPU: return "GPU";
      case VariantCode::OMP: return "OpenMP";
    }
    LEGATE_ABORT("Unknown variant code: " + std::to_string(to_underlying(variant_code)));
    return "";
  }();

  auto task_index = strategy.task_index();

  if (!task_index.has_value()) {
    return;
  }

  log_legate().print() << task_name << ' ' << variant_str << " task [" << provenance
                       << "], pt = " << *task_index;
}

template <bool WRITE>
[[nodiscard]] SmallVector<InternalSharedPtr<PhysicalStore>> map_stores(
  Span<const TaskStoreArg> store_args,
  const legate::mapping::detail::Machine& machine,
  const ParallelPolicy& parallel_policy,
  const Strategy& strategy,
  const Domain& launch_domain,
  const DomainPoint& task_index,
  DataMovementRequests* data_requests)
{
  SmallVector<InternalSharedPtr<PhysicalStore>> result;

  result.reserve(store_args.size());
  data_requests->reserve_copy_ins(store_args.size());
  for (auto&& [store, variable] : store_args) {
    auto&& partition                  = strategy[*variable];
    auto&& [physical_store, subsumed] = store->map(partition,
                                                   strategy.find_store_projection(*variable),
                                                   launch_domain,
                                                   task_index,
                                                   /*is_read=*/!WRITE,
                                                   machine,
                                                   data_requests);
    result.emplace_back(physical_store);
    if constexpr (WRITE) {
      // TODO(wonchanl): write accesses with slices are not supported yet
      LEGATE_CHECK(!store->transformed() || subsumed);
      if (!store->transformed()) {
        store->set_key_partition(machine, parallel_policy, partition);
      }
    }
  }
  return result;
}

[[nodiscard]] SmallVector<InternalSharedPtr<PhysicalStore>> map_reduction_stores(
  Span<const TaskStoreArg> store_args,
  Span<const std::pair<ReductionOpKind, bool>> reduction_ops,
  const legate::mapping::detail::Machine& machine,
  const Strategy& strategy,
  const Domain& launch_domain,
  const DomainPoint& task_index,
  DataMovementRequests* data_requests)
{
  SmallVector<InternalSharedPtr<PhysicalStore>> result;

  result.reserve(store_args.size());
  data_requests->reserve_reduction_requests(store_args.size());
  for (auto&& [store_arg, op_arg] : zip_equal(store_args, reduction_ops)) {
    auto&& [store, variable]             = store_arg;
    auto&& [op_kind, store_has_identity] = op_arg;
    auto&& partition                     = strategy[*variable];
    result.emplace_back(store->map_reduction(partition,
                                             strategy.find_store_projection(*variable),
                                             launch_domain,
                                             task_index,
                                             op_kind,
                                             store_has_identity,
                                             machine,
                                             data_requests));
  }
  return result;
}

[[nodiscard]] std::optional<TaskContext> make_task_context(const TaskBase* task,
                                                           VariantCode variant_code,
                                                           const Strategy& strategy,
                                                           DataMovementRequests* data_requests)
{
  auto launch_domain          = strategy.launch_domain();
  const auto maybe_task_index = strategy.task_index();

  if (!maybe_task_index.has_value()) {
    return std::nullopt;
  }

  auto&& task_index = *maybe_task_index;
  auto inputs       = map_stores<false>(task->inputs(),
                                        task->machine(),
                                        task->parallel_policy(),
                                        strategy,
                                        launch_domain,
                                        task_index,
                                        data_requests);
  auto outputs      = map_stores<true>(task->outputs(),
                                       task->machine(),
                                       task->parallel_policy(),
                                       strategy,
                                       launch_domain,
                                       task_index,
                                       data_requests);
  auto&& reductions = map_reduction_stores(task->reductions(),
                                           task->reduction_ops(),
                                           task->machine(),
                                           strategy,
                                           launch_domain,
                                           task_index,
                                           data_requests);

  return std::make_optional<TaskContext>(variant_code,
                                         launch_domain,
                                         task_index,
                                         task,
                                         std::move(inputs),
                                         std::move(outputs),
                                         std::move(reductions),
                                         SmallVector<InternalSharedPtr<Scalar>>{task->scalars()});
}

}  // namespace

void LogicalTask::launch_task_(Strategy* p_strategy)
{
  const auto get_task_name = [&] { return library().get_task_name(local_task_id()); };
  const auto variant_code  = machine().preferred_variant();

  show_progress(*p_strategy, get_task_name(), variant_code, provenance().as_string_view());

  DataMovementRequests data_requests{};

  auto maybe_ctx = make_task_context(this, variant_code, *p_strategy, &data_requests);

  if (!maybe_ctx.has_value() && data_requests.empty()) {
    // TODO(wonchanl): Ranks cannot exit early even when no tasks are assigned to them because they
    // still need to participat in the transfers
    return;
  }

  auto execute_task = [&] {
    data_requests.perform_copy_ins(variant_code);
    data_requests.perform_data_sync();

    variant_info_().body(legate::TaskContext{&*maybe_ctx});

    data_requests.perform_reductions(variant_code);
  };

  if (variant_code != VariantCode::GPU) {
    execute_task();
    return;
  }

  const cuda::detail::AutoPrimaryContext ctx{Runtime::get_runtime().get_current_cuda_device()};

  execute_task();
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
                                        std::int32_t redop_kind,
                                        bool store_has_identity)
{
  const auto* symb = find_or_declare_partition(store);
  add_reduction(std::move(store), redop_kind, symb, store_has_identity);
  return symb;
}

void AutoTask::add_input(InternalSharedPtr<LogicalStore> store, const Variable* partition_symbol)
{
  if (store->unbound()) {
    throw TracedException<std::invalid_argument>{"Unbound arrays cannot be used as input"};
  }

  auto& arg = inputs_.emplace_back(std::move(store), partition_symbol);

  record_partition_(partition_symbol, arg.store);
}

void AutoTask::add_output(InternalSharedPtr<LogicalStore> store, const Variable* partition_symbol)
{
  auto& arg = outputs_.emplace_back(std::move(store), partition_symbol);

  record_partition_(partition_symbol, arg.store);
}

void AutoTask::add_reduction(InternalSharedPtr<LogicalStore> store,
                             std::int32_t redop_kind,
                             const Variable* partition_symbol,
                             bool store_has_identity)
{
  if (store->unbound()) {
    throw TracedException<std::invalid_argument>{"Unbound store cannot be used for reductions"};
  }

  LEGATE_ASSERT(!store->type()->variable_size());

  reduction_ops_.emplace_back(static_cast<ReductionOpKind>(redop_kind), store_has_identity);

  auto& arg = reductions_.emplace_back(std::move(store), partition_symbol);

  record_partition_(partition_symbol, arg.store);
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

  add_store_(inputs_, store, create_no_partition());
}

void ManualTask::add_output(InternalSharedPtr<LogicalStore> store)
{
  add_store_(outputs_, store, create_no_partition());
}

void ManualTask::add_reduction(InternalSharedPtr<LogicalStore> store,
                               std::int32_t redop_kind,
                               bool store_has_identity)
{
  if (store->unbound()) {
    throw TracedException<std::invalid_argument>{"Unbound stores cannot be used for reduction"};
  }

  add_store_(reductions_, store, create_no_partition());

  reduction_ops_.emplace_back(static_cast<ReductionOpKind>(redop_kind), store_has_identity);
}

void ManualTask::add_input(const InternalSharedPtr<LogicalStorePartition>& store_partition,
                           std::optional<SymbolicPoint> projection,
                           bool is_key_partition)
{
  add_store_(inputs_,
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
  add_store_(outputs_,
             store_partition->store(),
             store_partition->partition(),
             std::move(projection),
             is_key_partition);
}

void ManualTask::add_reduction(const InternalSharedPtr<LogicalStorePartition>& store_partition,
                               std::int32_t redop_kind,
                               std::optional<SymbolicPoint> projection,
                               bool is_key_partition,
                               bool store_has_identity)
{
  add_store_(reductions_,
             store_partition->store(),
             store_partition->partition(),
             std::move(projection),
             is_key_partition);

  reduction_ops_.emplace_back(static_cast<ReductionOpKind>(redop_kind), store_has_identity);
}

void ManualTask::add_store_(SmallVector<TaskStoreArg>& store_args,
                            const InternalSharedPtr<LogicalStore>& store,
                            InternalSharedPtr<Partition> partition,
                            std::optional<SymbolicPoint> projection,
                            bool is_key_partition)
{
  const auto* partition_symbol = declare_partition();

  store_args.emplace_back(store, partition_symbol);

  record_partition_(partition_symbol, store);

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
