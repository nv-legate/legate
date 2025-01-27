/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <legate/runtime/detail/runtime.h>

#include <legate/comm/detail/coll.h>
#include <legate/comm/detail/comm.h>
#include <legate/cuda/detail/cuda_driver_api.h>
#include <legate/data/detail/array_tasks.h>
#include <legate/data/detail/external_allocation.h>
#include <legate/data/detail/logical_array.h>
#include <legate/data/detail/logical_region_field.h>
#include <legate/data/detail/logical_store.h>
#include <legate/experimental/io/detail/task.h>
#include <legate/mapping/detail/core_mapper.h>
#include <legate/mapping/detail/default_mapper.h>
#include <legate/mapping/detail/machine.h>
#include <legate/mapping/detail/mapping.h>
#include <legate/mapping/mapping.h>
#include <legate/operation/detail/attach.h>
#include <legate/operation/detail/copy.h>
#include <legate/operation/detail/discard.h>
#include <legate/operation/detail/execution_fence.h>
#include <legate/operation/detail/fill.h>
#include <legate/operation/detail/gather.h>
#include <legate/operation/detail/index_attach.h>
#include <legate/operation/detail/mapping_fence.h>
#include <legate/operation/detail/reduce.h>
#include <legate/operation/detail/release_region_field.h>
#include <legate/operation/detail/scatter.h>
#include <legate/operation/detail/scatter_gather.h>
#include <legate/operation/detail/task.h>
#include <legate/operation/detail/task_launcher.h>
#include <legate/partitioning/detail/constraint.h>
#include <legate/partitioning/detail/partitioner.h>
#include <legate/partitioning/detail/partitioning_tasks.h>
#include <legate/runtime/detail/argument_parsing.h>
#include <legate/runtime/detail/config.h>
#include <legate/runtime/detail/field_manager.h>
#include <legate/runtime/detail/library.h>
#include <legate/runtime/detail/shard.h>
#include <legate/runtime/resource.h>
#include <legate/runtime/runtime.h>
#include <legate/runtime/scope.h>
#include <legate/task/detail/legion_task.h>
#include <legate/task/detail/legion_task_body.h>
#include <legate/task/detail/task.h>
#include <legate/task/task_context.h>
#include <legate/task/variant_options.h>
#include <legate/type/detail/type_info.h>
#include <legate/utilities/detail/enumerate.h>
#include <legate/utilities/detail/env.h>
#include <legate/utilities/detail/env_defaults.h>
#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/detail/linearize.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/detail/tuple.h>
#include <legate/utilities/hash.h>
#include <legate/utilities/machine.h>
#include <legate/utilities/scope_guard.h>

#include <realm/cuda/cuda_module.h>
#include <realm/network.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <cstdlib>
#include <iostream>
#include <mappers/logging_wrapper.h>
#include <regex>
#include <set>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace legate::detail {

Logger& log_legate()
{
  static Logger log{"legate"};

  return log;
}

Logger& log_legate_partitioner()
{
  static Logger log{"legate.partitioner"};

  return log;
}

namespace {

// This is the unique string name for our library which can be used from both C++ and Python to
// generate IDs
constexpr std::string_view CORE_LIBRARY_NAME = "legate.core";
constexpr const char* const TOPLEVEL_NAME    = "Legate Core Toplevel Task";

}  // namespace

Runtime::Runtime()
  : legion_runtime_{Legion::Runtime::get_runtime()},
    window_size_{LEGATE_WINDOW_SIZE.get(LEGATE_WINDOW_SIZE_DEFAULT, LEGATE_WINDOW_SIZE_TEST)},
    field_reuse_freq_{
      LEGATE_FIELD_REUSE_FREQ.get(LEGATE_FIELD_REUSE_FREQ_DEFAULT, LEGATE_FIELD_REUSE_FREQ_TEST)},
    field_reuse_size_{local_machine().calculate_field_reuse_size()},
    force_consensus_match_{LEGATE_CONSENSUS.get(LEGATE_CONSENSUS_DEFAULT, LEGATE_CONSENSUS_TEST)}
{
}

Library* Runtime::create_library(
  std::string_view library_name,
  const ResourceConfig& config,
  std::unique_ptr<mapping::Mapper> mapper,
  // clang-tidy complains that this is not moved (but clearly, it is). I suspect this is due to
  // a tiny little footnote in try_emplace():
  //
  // ...unlike [unordered_map::]insert() or [unordered_map::]emplace(), these functions do not
  // move from rvalue arguments if the insertion does not happen...
  //
  // So probably there is a branch that clang-tidy sees down in try_emplace() that does
  // this. But clang-tidy seemingly does not realize that the emplacement always happens,
  // because we first check that the key does not yet exist.
  //
  // NOLINTNEXTLINE(performance-unnecessary-value-param)
  std::map<VariantCode, VariantOptions> default_options)
{
  if (libraries_.find(library_name) != libraries_.end()) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Library {} already exists", library_name)};
  }

  log_legate().debug() << "Library " << library_name << " is created";
  if (nullptr == mapper) {
    mapper = std::make_unique<mapping::detail::DefaultMapper>();
  }

  return &libraries_
            .try_emplace(std::string{library_name},
                         Library::ConstructKey{},
                         std::string{library_name},
                         config,
                         std::move(mapper),
                         std::move(default_options))
            .first->second;
}

namespace {

template <typename LibraryMapT>
[[nodiscard]] auto find_library_impl(LibraryMapT& libraries,
                                     std::string_view library_name,
                                     bool can_fail)
  -> std::conditional_t<std::is_const_v<LibraryMapT>, const Library*, Library*>
{
  const auto finder = libraries.find(library_name);

  if (libraries.end() == finder) {
    if (!can_fail) {
      throw TracedException<std::out_of_range>{
        fmt::format("Library {} does not exist", library_name)};
    }
    return {};
  }
  return &finder->second;
}

}  // namespace

const Library* Runtime::find_library(std::string_view library_name, bool can_fail /*=false*/) const
{
  return find_library_impl(libraries_, library_name, can_fail);
}

Library* Runtime::find_library(std::string_view library_name, bool can_fail /*=false*/)
{
  return find_library_impl(libraries_, library_name, can_fail);
}

Library* Runtime::find_or_create_library(
  std::string_view library_name,
  const ResourceConfig& config,
  std::unique_ptr<mapping::Mapper> mapper,
  const std::map<VariantCode, VariantOptions>& default_options,
  bool* created)
{
  auto result = find_library(library_name, true /*can_fail*/);

  if (result) {
    if (created) {
      *created = false;
    }
    return result;
  }
  result = create_library(std::move(library_name), config, std::move(mapper), default_options);
  if (created != nullptr) {
    *created = true;
  }
  return result;
}

void Runtime::record_reduction_operator(std::uint32_t type_uid,
                                        std::int32_t op_kind,
                                        GlobalRedopID legion_op_id)
{
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Record reduction op (type_uid: " << type_uid
                         << ", op_kind: " << op_kind
                         << ", legion_op_id: " << fmt::underlying(legion_op_id) << ")";
  }

  const auto inserted = reduction_ops_.try_emplace({type_uid, op_kind}, legion_op_id).second;

  if (!inserted) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Reduction op {} already exists for type {}", op_kind, type_uid)};
  }
}

GlobalRedopID Runtime::find_reduction_operator(std::uint32_t type_uid, std::int32_t op_kind) const
{
  const auto finder = reduction_ops_.find({type_uid, op_kind});

  if (reduction_ops_.end() == finder) {
    if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
      log_legate().debug() << "Can't find reduction op (type_uid: " << type_uid
                           << ", op_kind: " << op_kind << ")";
    }
    throw TracedException<std::invalid_argument>{
      fmt::format("Reduction op {} does not exist for type {}", op_kind, type_uid)};
  }
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Found reduction op " << fmt::underlying(finder->second)
                         << " (type_uid: " << type_uid << ", op_kind: " << op_kind << ")";
  }
  return finder->second;
}

void Runtime::initialize(Legion::Context legion_context)
{
  if (initialized()) {
    if (get_legion_context() == legion_context) {
      static_assert(std::is_pointer_v<Legion::Context>);
      LEGATE_CHECK(legion_context != nullptr);
      // OK to call initialize twice if it's the same context.
      return;
    }
    throw TracedException<std::runtime_error>{"Legate runtime has already been initialized"};
  }

  LEGATE_SCOPE_FAIL(
    // de-initialize everything in reverse order
    Config::has_socket_mem = false;
    scope_                 = Scope{};
    partition_manager_.reset();
    communicator_manager_.reset();
    field_manager_.reset();
    core_library_ = nullptr;
    comm::coll::finalize();
    cu_mod_manager_.reset();
    cu_driver_.reset();
    legion_context_ = {};
    initialized_    = false;);

  initialized_    = true;
  legion_context_ = std::move(legion_context);
  cu_driver_      = make_internal_shared<cuda::detail::CUDADriverAPI>();
  cu_mod_manager_.emplace(cu_driver_);
  comm::coll::init();
  field_manager_ = consensus_match_required() ? std::make_unique<ConsensusMatchingFieldManager>()
                                              : std::make_unique<FieldManager>();
  communicator_manager_.emplace();
  partition_manager_.emplace();
  static_cast<void>(scope_.exchange_machine(create_toplevel_machine()));

  Config::has_socket_mem = local_machine_.has_socket_memory();
  comm::register_builtin_communicator_factories(core_library());
}

std::pair<mapping::detail::Machine, const VariantInfo&> Runtime::slice_machine_for_task_(
  const TaskInfo& info) const
{
  const auto& machine = get_machine();
  auto sliced         = machine.only_if([&](mapping::TaskTarget t) {
    return info.find_variant(mapping::detail::to_variant_code(t)).has_value();
  });

  if (sliced.empty()) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Task {} does not have any valid variant for the current "
                  "machine configuration {}",
                  info,
                  machine)};
  }

  const auto variant = info.find_variant(sliced.preferred_variant());

  LEGATE_ASSERT(variant.has_value());
  return {sliced, *variant};  // NOLINT(bugprone-unchecked-optional-access)
}

// This function should be moved to the library context
InternalSharedPtr<AutoTask> Runtime::create_task(const Library* library, LocalTaskID task_id)
{
  auto&& [machine, vinfo] = slice_machine_for_task_(*library->find_task(task_id));
  auto task               = make_internal_shared<AutoTask>(
    library, vinfo, task_id, current_op_id_(), scope().priority(), std::move(machine));
  increment_op_id_();
  return task;
}

InternalSharedPtr<ManualTask> Runtime::create_task(const Library* library,
                                                   LocalTaskID task_id,
                                                   const Domain& launch_domain)
{
  if (launch_domain.empty()) {
    throw TracedException<std::invalid_argument>{"Launch domain must not be empty"};
  }
  auto&& [machine, vinfo] = slice_machine_for_task_(*library->find_task(task_id));
  auto task               = make_internal_shared<ManualTask>(library,
                                               vinfo,
                                               task_id,
                                               launch_domain,
                                               current_op_id_(),
                                               scope().priority(),
                                               std::move(machine));
  increment_op_id_();
  return task;
}

void Runtime::issue_copy(InternalSharedPtr<LogicalStore> target,
                         InternalSharedPtr<LogicalStore> source,
                         std::optional<std::int32_t> redop)
{
  submit(make_internal_shared<Copy>(std::move(target),
                                    std::move(source),
                                    current_op_id_(),
                                    scope().priority(),
                                    get_machine(),
                                    redop));
  increment_op_id_();
}

void Runtime::issue_gather(InternalSharedPtr<LogicalStore> target,
                           InternalSharedPtr<LogicalStore> source,
                           InternalSharedPtr<LogicalStore> source_indirect,
                           std::optional<std::int32_t> redop)
{
  submit(make_internal_shared<Gather>(std::move(target),
                                      std::move(source),
                                      std::move(source_indirect),
                                      current_op_id_(),
                                      scope().priority(),
                                      get_machine(),
                                      redop));
  increment_op_id_();
}

void Runtime::issue_scatter(InternalSharedPtr<LogicalStore> target,
                            InternalSharedPtr<LogicalStore> target_indirect,
                            InternalSharedPtr<LogicalStore> source,
                            std::optional<std::int32_t> redop)
{
  submit(make_internal_shared<Scatter>(std::move(target),
                                       std::move(target_indirect),
                                       std::move(source),
                                       current_op_id_(),
                                       scope().priority(),
                                       get_machine(),
                                       redop));
  increment_op_id_();
}

void Runtime::issue_scatter_gather(InternalSharedPtr<LogicalStore> target,
                                   InternalSharedPtr<LogicalStore> target_indirect,
                                   InternalSharedPtr<LogicalStore> source,
                                   InternalSharedPtr<LogicalStore> source_indirect,
                                   std::optional<std::int32_t> redop)
{
  submit(make_internal_shared<ScatterGather>(std::move(target),
                                             std::move(target_indirect),
                                             std::move(source),
                                             std::move(source_indirect),
                                             current_op_id_(),
                                             scope().priority(),
                                             get_machine(),
                                             redop));
  increment_op_id_();
}

void Runtime::issue_fill(const InternalSharedPtr<LogicalArray>& lhs,
                         InternalSharedPtr<LogicalStore> value)
{
  if (lhs->kind() != ArrayKind::BASE) {
    throw TracedException<std::runtime_error>{
      "Fills on list or struct arrays are not supported yet"};
  }

  if (value->type()->code == Type::Code::NIL) {
    if (!lhs->nullable()) {
      throw TracedException<std::invalid_argument>{
        "Non-nullable arrays cannot be filled with null"};
    }
    issue_fill(lhs->data(), Scalar{lhs->type()});
    issue_fill(lhs->null_mask(), Scalar{false});
    return;
  }

  issue_fill(lhs->data(), std::move(value));
  if (!lhs->nullable()) {
    return;
  }
  issue_fill(lhs->null_mask(), Scalar{true});
}

void Runtime::issue_fill(const InternalSharedPtr<LogicalArray>& lhs, Scalar value)
{
  if (lhs->kind() != ArrayKind::BASE) {
    throw TracedException<std::runtime_error>{
      "Fills on list or struct arrays are not supported yet"};
  }

  if (value.type()->code == Type::Code::NIL) {
    if (!lhs->nullable()) {
      throw TracedException<std::invalid_argument>{
        "Non-nullable arrays cannot be filled with null"};
    }
    issue_fill(lhs->data(), Scalar{lhs->type()});
    issue_fill(lhs->null_mask(), Scalar{false});
    return;
  }

  issue_fill(lhs->data(), std::move(value));
  if (!lhs->nullable()) {
    return;
  }
  issue_fill(lhs->null_mask(), Scalar{true});
}

void Runtime::issue_fill(InternalSharedPtr<LogicalStore> lhs, InternalSharedPtr<LogicalStore> value)
{
  if (lhs->unbound()) {
    throw TracedException<std::invalid_argument>{"Fill lhs must be a normal store"};
  }
  if (!value->has_scalar_storage()) {
    throw TracedException<std::invalid_argument>{"Fill value should be a Future-back store"};
  }

  submit(make_internal_shared<Fill>(
    std::move(lhs), std::move(value), current_op_id_(), scope().priority(), get_machine()));
  increment_op_id_();
}

void Runtime::issue_fill(InternalSharedPtr<LogicalStore> lhs, Scalar value)
{
  if (lhs->unbound()) {
    throw TracedException<std::invalid_argument>{"Fill lhs must be a normal store"};
  }

  submit(make_internal_shared<Fill>(
    std::move(lhs), std::move(value), current_op_id_(), scope().priority(), get_machine()));
  increment_op_id_();
}

void Runtime::tree_reduce(const Library* library,
                          LocalTaskID task_id,
                          InternalSharedPtr<LogicalStore> store,
                          InternalSharedPtr<LogicalStore> out_store,
                          std::int32_t radix)
{
  if (store->dim() != 1) {
    throw TracedException<std::runtime_error>{"Multi-dimensional stores are not supported"};
  }

  auto&& [machine, _] = slice_machine_for_task_(*library->find_task(task_id));

  submit(make_internal_shared<Reduce>(library,
                                      std::move(store),
                                      std::move(out_store),
                                      task_id,
                                      current_op_id_(),
                                      radix,
                                      scope().priority(),
                                      std::move(machine)));
  increment_op_id_();
}

namespace {
// OffloadTo is an empty task that runs with Read/Write permissions on its data,
// so that the data ends up getting copied to the target memory. Additionally, we
// modify the core-mapper to map the data to the specified target memory in
// `Runtime::offload_to()` for this task. This invalidates any other copies of the
// data in other memories and thus, frees up space.
class OffloadTo : public LegateTask<OffloadTo> {
 public:
  static constexpr auto TASK_ID = LocalTaskID{CoreTask::OFFLOAD_TO};

  // Task body left empty because there is no computation to do. This task
  // triggers a data movement because of its R/W privileges
  static void cpu_variant(legate::TaskContext) {}

  static void gpu_variant(legate::TaskContext) {}

  static void omp_variant(legate::TaskContext) {}  // namespace
};

}  // namespace

void Runtime::offload_to(mapping::StoreTarget target_mem,
                         const InternalSharedPtr<LogicalArray>& array)
{
  const auto target_proc = mapping::detail::get_matching_task_target(target_mem);

  switch (target_proc) {
    case mapping::TaskTarget::GPU: {
      if constexpr (!LEGATE_DEFINED(LEGATE_USE_CUDA)) {
        throw TracedException<std::invalid_argument>{
          fmt::format("Cannot offload to {} without GPU/CUDA Support", target_mem)};
      }
      break;
    }
    case mapping::TaskTarget::CPU: [[fallthrough]];
    case mapping::TaskTarget::OMP: break;
  }

  const auto scope = legate::Scope{legate::mapping::Machine{get_machine()}.only(target_proc)};
  auto task        = create_task(core_library(), OffloadTo::TASK_ID);

  task->add_scalar_arg(legate::Scalar{target_mem}.impl());

  // Ask for Read/Write privileges so that data is copied to the target_mem and
  // invalidated in other memories if any other copies exist.
  std::ignore = task->add_input(array);
  std::ignore = task->add_output(array);

  submit(std::move(task));
  // A mapping fence is issued here to prevent subsequent tasks from being mapped
  // and possibly creating new allocations on target_mem before the offload task
  // has had a chance to run. The fence is necessary because downstream tasks may
  // not have any data dependencies with the offload task and therefore, have no
  // guarantees on their mapping order.
  issue_mapping_fence();
}  // namespace

void Runtime::flush_scheduling_window()
{
  if (operations_.empty()) {
    return;
  }

  schedule_(std::move(operations_));
  operations_.clear();
}

void Runtime::submit(InternalSharedPtr<Operation> op)
{
  if constexpr (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_legate().debug() << op->to_string(true /*show_provenance*/) << " submitted";
  }

  // Special case for internal operations
  if (op->is_internal()) {
    operations_.emplace_back(std::move(op));
    if (operations_.size() >= window_size_) {
      flush_scheduling_window();
    }
    return;
  }

  op->validate();
  auto& submitted = operations_.emplace_back(std::move(op));
  if (submitted->needs_flush() || operations_.size() >= window_size_) {
    flush_scheduling_window();
  }
}

/*static*/ void Runtime::launch_immediately(const InternalSharedPtr<Operation>& op)
{
  // This function shouldn't be called if the operation needs to go through the full pipeline
  LEGATE_ASSERT(!op->needs_partitioning());

  op->launch();
}

void Runtime::schedule_(std::vector<InternalSharedPtr<Operation>>&& operations)
{
  // Move into temporary to "complete" the move from the caller side.
  const auto ops = std::move(operations);

  for (auto it = ops.begin(); it != ops.end(); ++it) {
    if constexpr (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
      log_legate().debug() << (*it)->to_string(true /*show_provenance*/) << " launched";
    }

    if (!(*it)->needs_partitioning()) {
      (*it)->launch();
      continue;
    }

    // TODO(wonchanl): We need the side effect from the launch calls to get key partitions set
    // correctly. In the future, the partitioner should manage key partitions.
    const auto strategy = Partitioner{{it, it + 1}}.partition_stores();
    (*it)->launch(strategy.get());
  }
}

InternalSharedPtr<LogicalArray> Runtime::create_array(const InternalSharedPtr<Shape>& shape,
                                                      InternalSharedPtr<Type> type,
                                                      bool nullable,
                                                      bool optimize_scalar)
{
  // TODO(wonchanl): We should be able to control colocation of fields for struct types,
  // instead of special-casing rect types here
  if (Type::Code::STRUCT == type->code && !is_rect_type(type)) {
    return create_struct_array_(shape, std::move(type), nullable, optimize_scalar);
  }

  if (type->variable_size()) {
    if (shape->ndim() != 1) {
      throw TracedException<std::invalid_argument>{"List/string arrays can only be 1D"};
    }

    auto elem_type  = Type::Code::STRING == type->code
                        ? int8()
                        : dynamic_cast<const detail::ListType&>(*type).element_type();
    auto descriptor = create_base_array_(shape, rect_type(1), nullable, optimize_scalar);
    auto vardata    = create_array(make_internal_shared<Shape>(1),
                                std::move(elem_type),
                                false /*nullable*/,
                                false /*optimize_scalar*/);

    return make_internal_shared<ListLogicalArray>(
      std::move(type), std::move(descriptor), std::move(vardata));
  }
  return create_base_array_(shape, std::move(type), nullable, optimize_scalar);
}

InternalSharedPtr<LogicalArray> Runtime::create_array_like(
  const InternalSharedPtr<LogicalArray>& array, InternalSharedPtr<Type> type)
{
  if (Type::Code::STRUCT == type->code || type->variable_size()) {
    throw TracedException<std::runtime_error>{
      "create_array_like doesn't support variable size types or struct types"};
  }
  if (array->unbound()) {
    return create_array(make_internal_shared<Shape>(array->dim()),
                        std::move(type),
                        array->nullable(),
                        false /*optimize_scalar*/);
  }

  const bool optimize_scalar = array->data()->has_scalar_storage();
  return create_array(array->shape(), std::move(type), array->nullable(), optimize_scalar);
}

InternalSharedPtr<LogicalArray> Runtime::create_list_array(
  InternalSharedPtr<Type> type,
  const InternalSharedPtr<LogicalArray>& descriptor,
  InternalSharedPtr<LogicalArray> vardata)
{
  if (Type::Code::STRING != type->code && Type::Code::LIST != type->code) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Expected a list type but got {}", *type)};
  }
  if (descriptor->unbound() || vardata->unbound()) {
    throw TracedException<std::invalid_argument>("Sub-arrays should not be unbound");
  }
  if (descriptor->dim() != 1 || vardata->dim() != 1) {
    throw TracedException<std::invalid_argument>("Sub-arrays should be 1D");
  }
  if (!is_rect_type(descriptor->type(), 1)) {
    throw TracedException<std::invalid_argument>{"Descriptor array does not have a 1D rect type"};
  }
  // If this doesn't hold, something bad happened (and will happen below)
  LEGATE_CHECK(!descriptor->nested());
  if (vardata->nullable()) {
    throw TracedException<std::invalid_argument>{"Vardata should not be nullable"};
  }

  auto elem_type = Type::Code::STRING == type->code
                     ? int8()
                     : dynamic_cast<const detail::ListType&>(*type).element_type();
  if (*vardata->type() != *elem_type) {
    throw TracedException<std::invalid_argument>{fmt::format(
      "Expected a vardata array of type {} but got {}", *elem_type, *(vardata->type()))};
  }

  return make_internal_shared<ListLogicalArray>(
    std::move(type), legate::static_pointer_cast<BaseLogicalArray>(descriptor), std::move(vardata));
}

InternalSharedPtr<StructLogicalArray> Runtime::create_struct_array_(
  const InternalSharedPtr<Shape>& shape,
  InternalSharedPtr<Type> type,
  bool nullable,
  bool optimize_scalar)
{
  std::vector<InternalSharedPtr<LogicalArray>> fields;
  const auto& st_type = dynamic_cast<const detail::StructType&>(*type);
  auto null_mask      = nullable ? create_store(shape, bool_(), optimize_scalar) : nullptr;

  fields.reserve(st_type.field_types().size());
  for (auto&& field_type : st_type.field_types()) {
    fields.emplace_back(create_array(shape, field_type, false, optimize_scalar));
  }
  return make_internal_shared<StructLogicalArray>(
    std::move(type), std::move(null_mask), std::move(fields));
}

InternalSharedPtr<BaseLogicalArray> Runtime::create_base_array_(InternalSharedPtr<Shape> shape,
                                                                InternalSharedPtr<Type> type,
                                                                bool nullable,
                                                                bool optimize_scalar)
{
  auto null_mask = nullable ? create_store(shape, bool_(), optimize_scalar) : nullptr;
  auto data      = create_store(std::move(shape), std::move(type), optimize_scalar);
  return make_internal_shared<BaseLogicalArray>(std::move(data), std::move(null_mask));
}

namespace {

void validate_store_shape(const InternalSharedPtr<Shape>& shape,
                          const InternalSharedPtr<Type>& type)
{
  if (shape->unbound()) {
    throw TracedException<std::invalid_argument>{
      "Shape of an unbound array or store cannot be used to create another store "
      "until the array or store is initialized by a task"};
  }
  if (type->variable_size()) {
    throw TracedException<std::invalid_argument>{"Store must have a fixed-size type"};
  }
}

}  // namespace

// Reserving the right to make this non-const in the future
// NOLINTNEXTLINE(readability-make-member-function-const)
InternalSharedPtr<LogicalStore> Runtime::create_store(InternalSharedPtr<Type> type,
                                                      std::uint32_t dim,
                                                      bool optimize_scalar)
{
  check_dimensionality_(dim);
  auto storage = make_internal_shared<detail::Storage>(make_internal_shared<Shape>(dim),
                                                       type->size(),
                                                       optimize_scalar,
                                                       get_provenance().as_string_view());
  return make_internal_shared<LogicalStore>(std::move(storage), std::move(type));
}  // namespace

// shape can be unbound in this function, so we shouldn't use the same validation as the other
// variants
// Reserving the right to make this non-const in the future
// NOLINTNEXTLINE(readability-make-member-function-const)
InternalSharedPtr<LogicalStore> Runtime::create_store(InternalSharedPtr<Shape> shape,
                                                      InternalSharedPtr<Type> type,
                                                      bool optimize_scalar /*=false*/)
{
  if (type->variable_size()) {
    throw TracedException<std::invalid_argument>{"Store must have a fixed-size type"};
  }
  check_dimensionality_(shape->ndim());
  auto storage = make_internal_shared<detail::Storage>(
    std::move(shape), type->size(), optimize_scalar, get_provenance().as_string_view());
  return make_internal_shared<LogicalStore>(std::move(storage), std::move(type));
}

// Reserving the right to make this non-const in the future
// NOLINTNEXTLINE(readability-make-member-function-const)
InternalSharedPtr<LogicalStore> Runtime::create_store(const Scalar& scalar,
                                                      InternalSharedPtr<Shape> shape)
{
  validate_store_shape(shape, scalar.type());
  if (shape->volume() != 1) {
    throw TracedException<std::invalid_argument>{"Scalar stores must have a shape of volume 1"};
  }
  auto future  = Legion::Future::from_untyped_pointer(scalar.data(), scalar.size());
  auto storage = make_internal_shared<detail::Storage>(
    std::move(shape), future, get_provenance().as_string_view());
  return make_internal_shared<detail::LogicalStore>(std::move(storage), scalar.type());
}

InternalSharedPtr<LogicalStore> Runtime::create_store(
  const InternalSharedPtr<Shape>& shape,
  InternalSharedPtr<Type> type,
  InternalSharedPtr<ExternalAllocation> allocation,
  InternalSharedPtr<mapping::detail::DimOrdering> ordering)
{
  validate_store_shape(shape, type);
  LEGATE_CHECK(allocation->ptr());
  if (allocation->size() < shape->volume() * type->size()) {
    throw TracedException<std::invalid_argument>{fmt::format(
      "External allocation of size {} is not big enough for a store of shape {} and type {}",
      allocation->size(),
      shape->extents(),
      *type)};
  }

  auto store = create_store(shape, std::move(type), false /*optimize_scalar*/);

  submit(make_internal_shared<Attach>(current_op_id_(),
                                      store->get_region_field(),
                                      store->dim(),
                                      std::move(allocation),
                                      std::move(ordering)));
  increment_op_id_();

  return store;
}

Runtime::IndexAttachResult Runtime::create_store(
  const InternalSharedPtr<Shape>& shape,
  const tuple<std::uint64_t>& tile_shape,
  InternalSharedPtr<Type> type,
  const std::vector<std::pair<legate::ExternalAllocation, tuple<std::uint64_t>>>& allocations,
  InternalSharedPtr<mapping::detail::DimOrdering> ordering)
{
  validate_store_shape(shape, type);

  auto type_size = type->size();
  auto store     = create_store(shape, std::move(type), false /*optimize_scalar*/);
  auto partition = partition_store_by_tiling(store, tile_shape);

  std::vector<Legion::LogicalRegion> subregions;
  std::vector<InternalSharedPtr<ExternalAllocation>> allocs;
  std::unordered_set<std::uint64_t> visited;
  const hasher<tuple<std::uint64_t>> hash_color{};
  subregions.reserve(allocations.size());
  allocs.reserve(allocations.size());
  visited.reserve(allocations.size());

  for (auto&& [idx, spec] : enumerate(allocations)) {
    auto&& [allocation, color] = spec;
    const auto color_hash      = hash_color(color);
    if (visited.find(color_hash) != visited.end()) {
      // If we're here, this color might have been seen in one of the previous iterations
      for (std::int64_t k = 0; k < idx; ++k) {
        if (allocations[k].second == color) {
          throw TracedException<std::invalid_argument>{
            fmt::format("Mulitple external allocations are found for color {}", color)};
        }
      }
      // If we're here, then we've just seen a fairly rare hash collision
    }
    visited.insert(color_hash);
    auto& alloc        = allocs.emplace_back(allocation.impl());
    auto substore      = partition->get_child_store(color);
    auto required_size = substore->volume() * type_size;

    LEGATE_ASSERT(alloc->ptr());

    if (!alloc->read_only()) {
      throw TracedException<std::invalid_argument>{"External allocations must be read-only"};
    }

    if (required_size > alloc->size()) {
      throw TracedException<std::invalid_argument>{
        fmt::format("Sub-store for color {}  requires the allocation to be at least {} bytes, but "
                    "the allocation is only {} bytes",
                    color,
                    required_size,
                    alloc->size())};
    }

    subregions.push_back(substore->get_region_field()->region());
  }

  submit(make_internal_shared<IndexAttach>(current_op_id_(),
                                           store->get_region_field(),
                                           store->dim(),
                                           std::move(subregions),
                                           std::move(allocs),
                                           std::move(ordering)));
  increment_op_id_();

  return {std::move(store), std::move(partition)};
}

void Runtime::prefetch_bloated_instances(InternalSharedPtr<LogicalStore> store,
                                         tuple<std::uint64_t> low_offsets,
                                         tuple<std::uint64_t> high_offsets,
                                         bool initialize)
{
  if (scope().machine()->count() <= 1) {
    log_legate().debug() << "Prefetching bloated instances is not necessary because there is only "
                            "one processor in the "
                            "scope";
    return;
  }
  if (initialize) {
    issue_fill(store, Scalar{store->type()});
  }

  auto arr  = LogicalArray::from_store(std::move(store));
  auto task = create_task(core_library_, legate::LocalTaskID{CoreTask::PREFETCH_BLOATED_INSTANCES});
  auto part1 = task->declare_partition();
  auto part2 = task->declare_partition();
  task->add_input(arr, part1);
  task->add_input(std::move(arr), part2);
  task->add_constraint(bloat(part2, part1, std::move(low_offsets), std::move(high_offsets)));
  submit(std::move(task));

  issue_mapping_fence();
}

void Runtime::check_dimensionality_(std::uint32_t dim)
{
  if (dim > LEGATE_MAX_DIM) {
    throw TracedException<std::out_of_range>{
      fmt::format("The maximum number of dimensions is {}, but a {}-D store is requested",
                  LEGION_MAX_DIM,
                  dim)};
  }
}

namespace {

class ExtractExceptionFn {
 public:
  [[nodiscard]] std::optional<ReturnedException> operator()(
    const Legion::Future& fut) const noexcept
  {
    if (auto exn = fut.get_result<ReturnedException>(); exn.raised()) {
      return {std::move(exn)};
    }
    return std::nullopt;
  }

  [[nodiscard]] std::optional<ReturnedException> operator()(
    ReturnedException& pending) const noexcept
  {
    if (pending.raised()) {
      return {std::move(pending)};
    }  // namespace
    return std::nullopt;
  }
};

}  // namespace

void Runtime::raise_pending_exception()
{
  std::optional<ReturnedException> found{};

  for (auto&& pending_exception : pending_exceptions_) {
    found = std::visit(ExtractExceptionFn{}, pending_exception);
    if (found.has_value()) {
      break;
    }
  }
  pending_exceptions_.clear();

  if (found.has_value()) {
    found->throw_exception();
  }
}

void Runtime::record_pending_exception(Legion::Future pending_exception)
{
  switch (scope().exception_mode()) {
    case ExceptionMode::IGNORED: return;
    case ExceptionMode::DEFERRED: {
      pending_exceptions_.emplace_back(std::move(pending_exception));
      break;
    }
    case ExceptionMode::IMMEDIATE: {
      if (auto&& exn = pending_exception.get_result<ReturnedException>(); exn.raised()) {
        exn.throw_exception();
      }
      break;
    }
  }
}

void Runtime::record_pending_exception(ReturnedException pending_exception)
{
  switch (scope().exception_mode()) {
    case ExceptionMode::IGNORED: return;
    case ExceptionMode::DEFERRED:
      pending_exceptions_.emplace_back(std::move(pending_exception));
      break;
    case ExceptionMode::IMMEDIATE:
      if (pending_exception.raised()) {
        pending_exception.throw_exception();
      }
      break;
  }
}

InternalSharedPtr<LogicalRegionField> Runtime::create_region_field(InternalSharedPtr<Shape> shape,
                                                                   std::uint32_t field_size)
{
  return field_manager()->allocate_field(std::move(shape), field_size);
}

InternalSharedPtr<LogicalRegionField> Runtime::import_region_field(InternalSharedPtr<Shape> shape,
                                                                   Legion::LogicalRegion region,
                                                                   Legion::FieldID field_id,
                                                                   std::uint32_t field_size)
{
  return field_manager()->import_field(std::move(shape), field_size, std::move(region), field_id);
}

void Runtime::attach_alloc_info(const InternalSharedPtr<LogicalRegionField>& rf,
                                // string_view is OK here because we pass the size along with it
                                std::string_view provenance)
{
  // It's safe to just attach the info to the FieldSpace+FieldID, since FieldSpaces are not shared
  // across RegionFields.
  if (provenance.empty()) {
    return;
  }
  get_legion_runtime()->attach_semantic_information(
    rf->region().get_field_space(),
    rf->field_id(),
    static_cast<Legion::SemanticTag>(CoreSemanticTag::ALLOC_INFO),
    provenance.data(),
    provenance.size(),
    /*is_mutable=*/true);
}

Legion::PhysicalRegion Runtime::map_region_field(Legion::LogicalRegion region,
                                                 Legion::FieldID field_id)
{
  Legion::RegionRequirement req{region, LEGION_READ_WRITE, LEGION_EXCLUSIVE, region};

  req.add_field(field_id);
  req.flags = LEGION_SUPPRESS_WARNINGS_FLAG;

  // TODO(wonchanl): We need to pass the metadata about logical store
  Legion::InlineLauncher launcher{req, mapper_id()};

  static_assert(std::is_same_v<decltype(launcher.provenance), std::string>,
                "Don't use to_string() below");
  launcher.provenance = get_provenance().to_string();
  return get_legion_runtime()->map_region(get_legion_context(), launcher);
}

void Runtime::remap_physical_region(Legion::PhysicalRegion pr)
{
  get_legion_runtime()->remap_region(
    get_legion_context(),
    std::move(pr),
    get_provenance().data()  // NOLINT(bugprone-suspicious-stringview-data-usage)
  );
  static_assert(std::is_same_v<decltype(get_provenance()), ZStringView>);
}

void Runtime::unmap_physical_region(Legion::PhysicalRegion pr)
{
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    std::vector<Legion::FieldID> fields;
    pr.get_fields(fields);
    LEGATE_CHECK(fields.size() == 1);
  }
  get_legion_runtime()->unmap_region(get_legion_context(), std::move(pr));
}

Legion::Future Runtime::detach(const Legion::PhysicalRegion& physical_region,
                               bool flush,
                               bool unordered)
{
  LEGATE_CHECK(physical_region.exists() && !physical_region.is_mapped());
  return get_legion_runtime()->detach_external_resource(
    get_legion_context(),
    physical_region,
    flush,
    unordered,
    get_provenance().data()  // NOLINT(bugprone-suspicious-stringview-data-usage)
  );
  static_assert(std::is_same_v<decltype(get_provenance()), ZStringView>);
}

Legion::Future Runtime::detach(const Legion::ExternalResources& external_resources,
                               bool flush,
                               bool unordered)
{
  LEGATE_CHECK(external_resources.exists());
  return get_legion_runtime()->detach_external_resources(
    get_legion_context(),
    external_resources,
    flush,
    unordered,
    get_provenance().data()  // NOLINT(bugprone-suspicious-stringview-data-usage)
  );
  static_assert(std::is_same_v<decltype(get_provenance()), ZStringView>);
}

bool Runtime::consensus_match_required() const
{
  return force_consensus_match_ || Legion::Machine::get_machine().get_address_space_count() > 1;
}

void Runtime::progress_unordered_operations()
{
  get_legion_runtime()->progress_unordered_operations(get_legion_context());
}

RegionManager* Runtime::find_or_create_region_manager(const Legion::IndexSpace& index_space)
{
  return &region_managers_.try_emplace(index_space, index_space).first->second;
}

const Legion::IndexSpace& Runtime::find_or_create_index_space(const tuple<std::uint64_t>& extents)
{
  if (extents.size() > LEGATE_MAX_DIM) {
    throw TracedException<std::out_of_range>{fmt::format(
      "Legate is configured with the maximum number of dimensions set to {}, but got a {}-D shape",
      LEGATE_MAX_DIM,
      extents.size())};
  }

  return find_or_create_index_space(to_domain(extents));
}

const Legion::IndexSpace& Runtime::find_or_create_index_space(const Domain& domain)
{
  LEGATE_CHECK(nullptr != get_legion_context());
  auto finder = cached_index_spaces_.find(domain);
  if (finder != cached_index_spaces_.end()) {
    return finder->second;
  }

  auto [it, _] = cached_index_spaces_.emplace(
    domain, get_legion_runtime()->create_index_space(get_legion_context(), domain));
  return it->second;
}

Legion::IndexPartition Runtime::create_restricted_partition(
  const Legion::IndexSpace& index_space,
  const Legion::IndexSpace& color_space,
  Legion::PartitionKind kind,
  const Legion::DomainTransform& transform,
  const Domain& extent)
{
  return get_legion_runtime()->create_partition_by_restriction(
    get_legion_context(), index_space, color_space, transform, extent, kind);
}

Legion::IndexPartition Runtime::create_weighted_partition(const Legion::IndexSpace& index_space,
                                                          const Legion::IndexSpace& color_space,
                                                          const Legion::FutureMap& weights)
{
  return get_legion_runtime()->create_partition_by_weights(
    get_legion_context(), index_space, weights, color_space);
}

Legion::IndexPartition Runtime::create_image_partition(
  const Legion::IndexSpace& index_space,
  const Legion::IndexSpace& color_space,
  const Legion::LogicalRegion& func_region,
  const Legion::LogicalPartition& func_partition,
  Legion::FieldID func_field_id,
  bool is_range,
  const mapping::detail::Machine& machine)
{
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Create image partition {index_space: " << index_space
                         << ", func_partition: " << func_partition
                         << ", func_field_id: " << func_field_id << ", is_range: " << is_range
                         << "}";
  }

  BufferBuilder buffer;
  machine.pack(buffer);
  buffer.pack<std::uint32_t>(get_sharding(machine, 0));
  buffer.pack<std::int32_t>(scope().priority());

  if (is_range) {
    return get_legion_runtime()->create_partition_by_image_range(get_legion_context(),
                                                                 index_space,
                                                                 func_partition,
                                                                 func_region,
                                                                 func_field_id,
                                                                 color_space,
                                                                 LEGION_COMPUTE_KIND,
                                                                 LEGION_AUTO_GENERATE_ID,
                                                                 mapper_id(),
                                                                 0,
                                                                 buffer.to_legion_buffer());
  }
  return get_legion_runtime()->create_partition_by_image(get_legion_context(),
                                                         index_space,
                                                         func_partition,
                                                         func_region,
                                                         func_field_id,
                                                         color_space,
                                                         LEGION_COMPUTE_KIND,
                                                         LEGION_AUTO_GENERATE_ID,
                                                         mapper_id(),
                                                         0,
                                                         buffer.to_legion_buffer());
}

Legion::IndexPartition Runtime::create_approximate_image_partition(
  const InternalSharedPtr<LogicalStore>& store,
  const InternalSharedPtr<Partition>& partition,
  const Legion::IndexSpace& index_space,
  bool sorted)
{
  LEGATE_ASSERT(partition->has_launch_domain());
  auto&& launch_domain = partition->launch_domain();
  auto output          = create_store(domain_type(), 1, true);
  auto task            = create_task(
    core_library(),
    LocalTaskID{sorted ? CoreTask::FIND_BOUNDING_BOX_SORTED : CoreTask::FIND_BOUNDING_BOX},
    launch_domain);

  task->add_input(create_store_partition(store, partition, std::nullopt), std::nullopt);
  task->add_output(output);
  // Directly launch the partitioning task, instead of going through the scheduling pipeline,
  // because this function is invoked only when the partition is immediately needed.
  launch_immediately(std::move(task));

  auto domains     = output->get_future_map();
  auto color_space = find_or_create_index_space(launch_domain);
  return get_legion_runtime()->create_partition_by_domain(
    get_legion_context(), index_space, domains, color_space);
}

Legion::FieldSpace Runtime::create_field_space()
{
  LEGATE_CHECK(nullptr != get_legion_context());
  return get_legion_runtime()->create_field_space(get_legion_context());
}

Legion::LogicalRegion Runtime::create_region(const Legion::IndexSpace& index_space,
                                             const Legion::FieldSpace& field_space)
{
  LEGATE_CHECK(nullptr != get_legion_context());
  return get_legion_runtime()->create_logical_region(
    get_legion_context(), index_space, field_space);
}

void Runtime::destroy_region(const Legion::LogicalRegion& logical_region, bool unordered)
{
  LEGATE_CHECK(nullptr != get_legion_context());
  get_legion_runtime()->destroy_logical_region(get_legion_context(), logical_region, unordered);
}

Legion::LogicalPartition Runtime::create_logical_partition(
  const Legion::LogicalRegion& logical_region, const Legion::IndexPartition& index_partition)
{
  LEGATE_CHECK(nullptr != get_legion_context());
  return get_legion_runtime()->get_logical_partition(
    get_legion_context(), logical_region, index_partition);
}

Legion::LogicalRegion Runtime::get_subregion(const Legion::LogicalPartition& partition,
                                             const Legion::DomainPoint& color)
{
  LEGATE_CHECK(nullptr != get_legion_context());
  return get_legion_runtime()->get_logical_subregion_by_color(
    get_legion_context(), partition, color);
}

Legion::LogicalRegion Runtime::find_parent_region(const Legion::LogicalRegion& region)
{
  auto result = region;
  while (get_legion_runtime()->has_parent_logical_partition(get_legion_context(), result)) {
    auto partition =
      get_legion_runtime()->get_parent_logical_partition(get_legion_context(), result);
    result = get_legion_runtime()->get_parent_logical_region(get_legion_context(), partition);
  }
  return result;
}

Legion::FieldID Runtime::allocate_field(const Legion::FieldSpace& field_space,
                                        Legion::FieldID field_id,
                                        std::size_t field_size)
{
  LEGATE_CHECK(nullptr != get_legion_context());
  auto allocator = get_legion_runtime()->create_field_allocator(get_legion_context(), field_space);
  return allocator.allocate_field(field_size, field_id);
}

Domain Runtime::get_index_space_domain(const Legion::IndexSpace& index_space)
{
  return get_legion_runtime()->get_index_space_domain(get_legion_context(), index_space);
}

namespace {

[[nodiscard]] Legion::DomainPoint delinearize_future_map_impl(const DomainPoint& point,
                                                              const Domain& domain,
                                                              const Domain& range)
{
  LEGATE_CHECK(range.dim == 1);
  DomainPoint result;
  result.dim = 1;
  result[0]  = static_cast<coord_t>(linearize(domain.lo(), domain.hi(), point));
  return result;
}

[[nodiscard]] Legion::DomainPoint reshape_future_map_impl(const DomainPoint& point,
                                                          const Domain& domain,
                                                          const Domain& range)
{
  // TODO(wonchanl): This reshaping preserves the mapping of tasks if both the producer and consumer
  // point tasks are linearized in the same way in the mapper, which is the case now.
  // If in the future we make the mapping from points in the launch domain to GPUs customizable,
  // we need to take that into account here as well.
  return delinearize(range.lo(), range.hi(), linearize(domain.lo(), domain.hi(), point));
}  // namespace

}  // namespace

Legion::FutureMap Runtime::delinearize_future_map(const Legion::FutureMap& future_map,
                                                  const Domain& new_domain)
{
  return get_legion_runtime()->transform_future_map(get_legion_context(),
                                                    future_map,
                                                    find_or_create_index_space(new_domain),
                                                    delinearize_future_map_impl);
}

Legion::FutureMap Runtime::reshape_future_map(const Legion::FutureMap& future_map,
                                              const Legion::Domain& new_domain)
{
  return get_legion_runtime()->transform_future_map(get_legion_context(),
                                                    future_map,
                                                    find_or_create_index_space(new_domain),
                                                    reshape_future_map_impl);
}

std::pair<Legion::PhaseBarrier, Legion::PhaseBarrier> Runtime::create_barriers(
  std::size_t num_tasks)
{
  auto arrival_barrier =
    get_legion_runtime()->create_phase_barrier(get_legion_context(), num_tasks);
  auto wait_barrier =
    get_legion_runtime()->advance_phase_barrier(get_legion_context(), arrival_barrier);
  return {arrival_barrier, wait_barrier};
}

void Runtime::destroy_barrier(Legion::PhaseBarrier barrier)
{
  get_legion_runtime()->destroy_phase_barrier(get_legion_context(), std::move(barrier));
}

Legion::Future Runtime::get_tunable(const Library& library, std::int64_t tunable_id)
{
  auto launcher =
    Legion::TunableLauncher{static_cast<Legion::TunableID>(tunable_id), mapper_id(), 0};
  const auto* mapper = library.get_mapper();

  launcher.arg = Legion::UntypedBuffer{
    mapper,
    sizeof(mapper)  // NOLINT(bugprone-sizeof-expression)
  };
  return get_legion_runtime()->select_tunable_value(get_legion_context(), launcher);
}

Legion::Future Runtime::dispatch(Legion::TaskLauncher& launcher,
                                 std::vector<Legion::OutputRequirement>& output_requirements)
{
  LEGATE_CHECK(nullptr != get_legion_context());
  return get_legion_runtime()->execute_task(get_legion_context(), launcher, &output_requirements);
}

Legion::FutureMap Runtime::dispatch(Legion::IndexTaskLauncher& launcher,
                                    std::vector<Legion::OutputRequirement>& output_requirements)
{
  LEGATE_CHECK(nullptr != get_legion_context());
  return get_legion_runtime()->execute_index_space(
    get_legion_context(), launcher, &output_requirements);
}

void Runtime::dispatch(const Legion::CopyLauncher& launcher)
{
  LEGATE_CHECK(nullptr != get_legion_context());
  get_legion_runtime()->issue_copy_operation(get_legion_context(), launcher);
}

void Runtime::dispatch(const Legion::IndexCopyLauncher& launcher)
{
  LEGATE_CHECK(nullptr != get_legion_context());
  get_legion_runtime()->issue_copy_operation(get_legion_context(), launcher);
}

void Runtime::dispatch(const Legion::FillLauncher& launcher)
{
  LEGATE_CHECK(nullptr != get_legion_context());
  get_legion_runtime()->fill_fields(get_legion_context(), launcher);
}

void Runtime::dispatch(const Legion::IndexFillLauncher& launcher)
{
  LEGATE_CHECK(nullptr != get_legion_context());
  get_legion_runtime()->fill_fields(get_legion_context(), launcher);
}

Legion::Future Runtime::extract_scalar(const Legion::Future& result,
                                       std::size_t offset,
                                       std::size_t size) const
{
  const auto& machine = get_machine();
  auto provenance     = get_provenance();
  auto variant        = static_cast<Legion::MappingTagID>(machine.preferred_variant());
  auto launcher       = TaskLauncher{
    core_library(), machine, provenance, LocalTaskID{CoreTask::EXTRACT_SCALAR}, variant};

  launcher.add_future(result);
  launcher.add_scalar(make_internal_shared<Scalar>(offset));
  launcher.add_scalar(make_internal_shared<Scalar>(size));
  return launcher.execute_single();
}

Legion::FutureMap Runtime::extract_scalar(const Legion::FutureMap& result,
                                          std::size_t offset,
                                          std::size_t size,
                                          const Legion::Domain& launch_domain) const
{
  const auto& machine = get_machine();
  auto provenance     = get_provenance();
  auto variant        = static_cast<Legion::MappingTagID>(machine.preferred_variant());
  auto launcher       = TaskLauncher{
    core_library(), machine, provenance, LocalTaskID{CoreTask::EXTRACT_SCALAR}, variant};

  launcher.add_future_map(result);
  launcher.add_scalar(make_internal_shared<Scalar>(offset));
  launcher.add_scalar(make_internal_shared<Scalar>(size));
  return launcher.execute(launch_domain);
}

Legion::Future Runtime::reduce_future_map(const Legion::FutureMap& future_map,
                                          GlobalRedopID reduction_op,
                                          const Legion::Future& init_value)
{
  return get_legion_runtime()->reduce_future_map(get_legion_context(),
                                                 future_map,
                                                 static_cast<Legion::ReductionOpID>(reduction_op),
                                                 false /*deterministic*/,
                                                 mapper_id(),
                                                 0 /*tag*/,
                                                 nullptr /*provenance*/,
                                                 init_value);
}

Legion::Future Runtime::reduce_exception_future_map(const Legion::FutureMap& future_map)
{
  const auto reduction_op =
    core_library()->get_reduction_op_id(LocalRedopID{CoreReductionOp::JOIN_EXCEPTION});

  return get_legion_runtime()->reduce_future_map(
    get_legion_context(),
    future_map,
    static_cast<Legion::ReductionOpID>(reduction_op),
    false /*deterministic*/,
    mapper_id(),
    static_cast<Legion::MappingTagID>(CoreMappingTag::JOIN_EXCEPTION));
}

void Runtime::issue_release_region_field(
  InternalSharedPtr<LogicalRegionField::PhysicalState> physical_state, bool unmap, bool unordered)
{
  submit(make_internal_shared<ReleaseRegionField>(
    current_op_id_(), std::move(physical_state), unmap, unordered));
  increment_op_id_();
}

void Runtime::issue_discard_field(const Legion::LogicalRegion& region, Legion::FieldID field_id)
{
  submit(make_internal_shared<Discard>(current_op_id_(), region, field_id));
  increment_op_id_();
}

void Runtime::issue_mapping_fence()
{
  submit(make_internal_shared<MappingFence>(current_op_id_()));
  increment_op_id_();
}

void Runtime::issue_execution_fence(bool block /*=false*/)
{
  // FIXME: This needs to be a Legate operation
  submit(make_internal_shared<ExecutionFence>(current_op_id_(), block));
  increment_op_id_();
  if (block) {
    // This will block on the execution fence
    flush_scheduling_window();
  }
}

InternalSharedPtr<LogicalStore> Runtime::get_timestamp(Timing::Precision precision)
{
  static auto empty_shape = make_internal_shared<Shape>(tuple<std::uint64_t>{});
  auto timestamp          = create_store(empty_shape, int64(), true /* optimize_scalar */);
  submit(make_internal_shared<Timing>(current_op_id_(), precision, timestamp));
  increment_op_id_();
  return timestamp;
}

void Runtime::begin_trace(std::uint32_t trace_id)
{
  flush_scheduling_window();
  get_legion_runtime()->begin_trace(get_legion_context(), trace_id);
}

void Runtime::end_trace(std::uint32_t trace_id)
{
  flush_scheduling_window();
  get_legion_runtime()->end_trace(get_legion_context(), trace_id);
}

InternalSharedPtr<mapping::detail::Machine> Runtime::create_toplevel_machine()
{
  const auto num_nodes = local_machine_.total_nodes;
  const auto num_gpus  = local_machine_.total_gpu_count();
  const auto num_omps  = local_machine_.total_omp_count();
  const auto num_cpus  = local_machine_.total_cpu_count();
  auto create_range    = [&num_nodes](std::uint32_t num_procs) {
    auto per_node_count = static_cast<std::uint32_t>(num_procs / num_nodes);
    return mapping::ProcessorRange{0, num_procs, per_node_count};
  };

  return make_internal_shared<mapping::detail::Machine>(
    std::map<mapping::TaskTarget, mapping::ProcessorRange>{
      {mapping::TaskTarget::GPU, create_range(num_gpus)},
      {mapping::TaskTarget::OMP, create_range(num_omps)},
      {mapping::TaskTarget::CPU, create_range(num_cpus)}});
}

const mapping::detail::Machine& Runtime::get_machine() const { return *scope().machine(); }

ZStringView Runtime::get_provenance() const { return scope().provenance(); }

Legion::ProjectionID Runtime::get_affine_projection(std::uint32_t src_ndim,
                                                    const proj::SymbolicPoint& point)
{
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Query affine projection {src_ndim: " << src_ndim
                         << ", point: " << point << "}";
  }

  if (proj::is_identity(src_ndim, point)) {
    if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
      log_legate().debug() << "Identity projection {src_ndim: " << src_ndim << ", point: " << point
                           << "}";
    }
    return 0;
  }

  auto key    = AffineProjectionDesc{src_ndim, point};
  auto finder = affine_projections_.find(key);
  if (affine_projections_.end() != finder) {
    return finder->second;
  }

  auto proj_id = core_library()->get_projection_id(next_projection_id_++);

  register_affine_projection_functor(src_ndim, point, proj_id);
  affine_projections_[key] = proj_id;

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Register affine projection " << proj_id << " {src_ndim: " << src_ndim
                         << ", point: " << point << "}";
  }

  return proj_id;
}

Legion::ProjectionID Runtime::get_delinearizing_projection(const tuple<std::uint64_t>& color_shape)
{
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Query delinearizing projection {color_shape: "
                         << color_shape.to_string() << "}";
  }

  auto finder = delinearizing_projections_.find(color_shape);
  if (delinearizing_projections_.end() != finder) {
    return finder->second;
  }

  auto proj_id = core_library()->get_projection_id(next_projection_id_++);

  register_delinearizing_projection_functor(color_shape, proj_id);
  delinearizing_projections_[color_shape] = proj_id;

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Register delinearizing projection " << proj_id
                         << "{color_shape: " << color_shape.to_string() << "}";
  }

  return proj_id;
}

Legion::ProjectionID Runtime::get_compound_projection(const tuple<std::uint64_t>& color_shape,
                                                      const proj::SymbolicPoint& point)
{
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Query compound projection {color_shape: " << color_shape.to_string()
                         << ", point: " << point << "}";
  }

  auto key    = CompoundProjectionDesc{color_shape, point};
  auto finder = compound_projections_.find(key);
  if (compound_projections_.end() != finder) {
    return finder->second;
  }

  auto proj_id = core_library()->get_projection_id(next_projection_id_++);

  register_compound_projection_functor(color_shape, point, proj_id);
  compound_projections_[key] = proj_id;

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Register compound projection " << proj_id
                         << " {color_shape: " << color_shape.to_string() << ", point: " << point
                         << "}";
  }

  return proj_id;
}

Legion::ShardingID Runtime::get_sharding(const mapping::detail::Machine& machine,
                                         Legion::ProjectionID proj_id)
{
  // If we're running on a single node, we don't need to generate sharding functors
  if (Realm::Network::max_node_id == 0) {
    return 0;
  }

  auto& proc_range = machine.processor_range();
  ShardingDesc key{proj_id, proc_range};

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Query sharding {proj_id: " << proj_id
                         << ", processor range: " << proc_range
                         << ", processor type: " << machine.preferred_target() << "}";
  }

  auto finder = registered_shardings_.find(key);
  if (finder != registered_shardings_.end()) {
    if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
      log_legate().debug() << "Found sharding " << finder->second;
    }
    return finder->second;
  }

  auto sharding_id = core_library()->get_sharding_id(next_sharding_id_++);
  registered_shardings_.insert({std::move(key), sharding_id});

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Create sharding " << sharding_id;
  }

  create_sharding_functor_using_projection(sharding_id, proj_id, proc_range);

  return sharding_id;
}

namespace {

void handle_realm_default_args()
{
  constexpr EnvironmentVariable<std::uint32_t> OMPI_COMM_WORLD_SIZE{"OMPI_COMM_WORLD_SIZE"};
  constexpr EnvironmentVariable<std::uint32_t> MV2_COMM_WORLD_SIZE{"MV2_COMM_WORLD_SIZE"};
  constexpr EnvironmentVariable<std::uint32_t> SLURM_NTASKS{"SLURM_NTASKS"};

  if (OMPI_COMM_WORLD_SIZE.get(/* default_value */ 1) == 1 &&
      MV2_COMM_WORLD_SIZE.get(/* default_value */ 1) == 1 &&
      SLURM_NTASKS.get(/* default_vaule */ 1) == 1) {
    constexpr EnvironmentVariable<std::string> REALM_DEFAULT_ARGS{"REALM_DEFAULT_ARGS"};
    std::stringstream ss;

    if (const auto existing_default_args = REALM_DEFAULT_ARGS.get()) {
      static const auto networks_re = std::regex{R"(\-ll:networks\s+\w+)", std::regex::optimize};

      // If Realm sees multiple networks arguments, with one of them being "none", (e.g.
      // "-ll:networks foo -ll:networks none", or even "-ll:networks none -ll:networks none"),
      // it balks with:
      //
      // "Cannot specify both 'none' and another value in -ll:networks"
      //
      // So we must strip away any existing -ll:networks arguments before we append our
      // -ll:networks argument.
      ss << std::regex_replace(existing_default_args->c_str(), networks_re, "");
    }
    ss << " -ll:networks none ";
    REALM_DEFAULT_ARGS.set(ss.str());
  }
}

}  // namespace

/*static*/ void Runtime::start()
{
  // Call as soon as possible, to ensure that any exceptions are pretty-printed
  static_cast<void>(install_terminate_handler());
  // Must populate this before we handle Legate args as it expects to read its values.
  Config::parse();
  if (!Legion::Runtime::has_runtime()) {
    handle_realm_default_args();

    int dummy_argc    = 0;
    char** dummy_argv = nullptr;

    Legion::Runtime::initialize(&dummy_argc, &dummy_argv, /*filter=*/false, /*parse=*/false);
    handle_legate_args();
  }

  // Do these after handle_legate_args()
  if (!LEGATE_DEFINED(LEGATE_USE_CUDA) && LEGATE_NEED_CUDA.get(/* default_value = */ false)) {
    throw TracedException<std::runtime_error>{
      "Legate was run with GPUs but was not built with GPU support. "
      "Please install Legate again with the \"--with-cuda\" flag"};
  }
  if (!LEGATE_DEFINED(LEGATE_USE_OPENMP) && LEGATE_NEED_OPENMP.get(/* default_value = */ false)) {
    throw TracedException<std::runtime_error>{
      "Legate was run with OpenMP enabled, but was not built with OpenMP support. "
      "Please install Legate again with the \"--with-openmp\" flag"};
  }
  if (!LEGATE_DEFINED(LEGATE_USE_NETWORK) && LEGATE_NEED_NETWORK.get(/* default_value = */ false)) {
    throw TracedException<std::runtime_error>{
      "Legate was run on multiple nodes but was not built with networking "
      "support. Please install Legate again with network support (e.g. \"--with-mpi\" or "
      "\"--with-gasnet\")"};
  }

  Legion::Runtime::perform_registration_callback(initialize_core_library_callback_,
                                                 true /*global*/);

  if (!Legion::Runtime::has_runtime()) {
    if (const auto result =
          Legion::Runtime::start(/*argc*/ 0, /*argv*/ nullptr, /*background=*/true)) {
      throw TracedException<std::runtime_error>{
        fmt::format("Legion Runtime failed to start, error code: {}", result)};
    }
  }

  // Get the runtime now that we've started it
  auto legion_runtime = Legion::Runtime::get_runtime();

  Legion::Context legion_context;
  // If the context already exists, that means that some other driver started the top-level task,
  // so here we just grab it to initialize the Legate runtime
  if (Legion::Runtime::has_context()) {
    legion_context = Legion::Runtime::get_context();
  } else {
    // Otherwise we  make this thread into an implicit top-level task
    legion_context = legion_runtime->begin_implicit_task(CoreTask::TOPLEVEL,
                                                         0 /*mapper id*/,
                                                         Processor::LOC_PROC,
                                                         TOPLEVEL_NAME,
                                                         true /*control replicable*/);
  }

  // We can now initialize the Legate runtime with the Legion context
  Runtime::get_runtime()->initialize(legion_context);
}

void Runtime::start_profiling_range()
{
  auto legion_runtime = get_legion_runtime();
  auto legion_context = Legion::Runtime::get_context();
  legion_runtime->start_profiling_range(legion_context);
}

void Runtime::stop_profiling_range(std::string_view provenance)
{
  auto legion_runtime = get_legion_runtime();
  auto legion_context = Legion::Runtime::get_context();
  legion_runtime->stop_profiling_range(legion_context, std::string{provenance}.c_str());
}

namespace {

class RuntimeManager {
 public:
  enum class State : std::uint8_t { UNINITIALIZED, INITIALIZED, FINALIZED };

  [[nodiscard]] Runtime* get();
  [[nodiscard]] State state() const noexcept;
  void reset() noexcept;

 private:
  State state_{State::UNINITIALIZED};
  std::optional<Runtime> rt_{};
};

Runtime* RuntimeManager::get()
{
  if (LEGATE_UNLIKELY(!rt_.has_value())) {
    if (state() == State::FINALIZED) {
      // Legion currently does not allow re-initialization after shutdown, so we need to track
      // this ourselves...
      throw TracedException<std::runtime_error>{
        "Legate runtime has been finalized, and cannot be re-initialized without restarting the "
        "program."};
    }  // namespace
    rt_.emplace();
    state_ = State::INITIALIZED;
  }  // namespace
  return &*rt_;
}  // namespace

RuntimeManager::State RuntimeManager::state() const noexcept { return state_; }

void RuntimeManager::reset() noexcept
{
  rt_.reset();
  state_ = State::FINALIZED;
}

RuntimeManager the_runtime{};

}  // namespace

/*static*/ Runtime* Runtime::get_runtime() { return the_runtime.get(); }

void Runtime::register_shutdown_callback(ShutdownCallback callback)
{
  callbacks_.emplace_back(std::move(callback));
}

std::int32_t Runtime::finish()
{
  if (!has_started()) {
    return 0;
  }

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Destroying Legate runtime...";
  }

  // NOLINTNEXTLINE(modernize-loop-convert) `callbacks_` can be modified by callbacks
  for (auto it = callbacks_.begin(); it != callbacks_.end(); ++it) {
    (*it)();
  }

  // Flush any outstanding operations before we tear down the runtime
  flush_scheduling_window();

  // Need a fence to make sure all client operations come before the subsequent clean-up tasks
  issue_execution_fence();

  // Destroy all communicators. This will likely launch some tasks for the clean-up.
  communicator_manager()->destroy();

  // Destroy all Legion handles used by Legate
  for (auto&& [_, region_manager] : region_managers_) {
    region_manager.destroy(true /*unordered*/);
  }
  for (auto&& [_, index_space] : cached_index_spaces_) {
    get_legion_runtime()->destroy_index_space(
      get_legion_context(), index_space, true /*unordered*/);
  }
  cached_index_spaces_.clear();

  // We're about to deallocate objects below, so let's block on all outstanding Legion operations
  issue_execution_fence(true);

  // Any STL containers holding Legion handles need to be cleared here, otherwise they cause
  // trouble when they get destroyed in the Legate runtime's destructor

  // We finally deallocate managers
  region_managers_.clear();
  field_manager_.reset();

  communicator_manager_.reset();
  partition_manager_.reset();
  scope_        = Scope{};
  core_library_ = nullptr;
  comm::coll::finalize();
  // Mappers get raw pointers to Libraries, so just in case any of the above launched residual
  // cleanup tasks, we issue another fence here before we clear the Libraries.
  issue_execution_fence(true);
  mapper_manager_.reset();
  libraries_.clear();
  // Reset this here and now (instead of waiting to destroy it when Runtime gets destroyed)
  // because the dtor of the module manager tries to get the runtime, which it should not do if
  // we are in the middle of self-destructing.
  cu_mod_manager_.reset();
  // This should be empty at this point, since the execution fence will ensure they are all
  // raised, but just in case, clear them. There is no hope of properly handling them now.
  pending_exceptions_.clear();
  initialized_ = false;

  // Mark that we are done excecuting the top-level task
  // After this call the context is no longer valid
  get_legion_runtime()->finish_implicit_task(std::exchange(legion_context_, nullptr));
  // The previous call is asynchronous so we still need to
  // wait for the shutdown of the runtime to complete
  const auto ret = Legion::Runtime::wait_for_shutdown();
  // Do NOT delete, move, re-order, or otherwise modify the following lines under ANY
  // circumstances.
  //
  // They must stay exactly as they are. the_runtime.reset() calls "delete this", and hence any
  // modification of the runtime object, or any of its derivatives hereafter is strictly
  // undefined behavior.
  //
  // BEGIN DO NOT MODIFY
  the_runtime.reset();
  return ret;
  // END DO NOT MODIFY
}  // namespace

namespace {

template <VariantCode variant_kind>
void extract_scalar_task(const void* args,
                         std::size_t arglen,
                         const void* /*userdata*/,
                         std::size_t /*userlen*/,
                         Legion::Processor p)
{
  // TODO(jfaibussowit)
  // This task should really be going through the proper channels instead of this Frankenstein.
  const Legion::Task* task;
  const std::vector<Legion::PhysicalRegion>* regions;
  Legion::Context legion_context;
  Legion::Runtime* runtime;
  Legion::Runtime::legion_task_preamble(args, arglen, p, task, regions, legion_context, runtime);

  show_progress(task, legion_context, runtime);

  auto legion_task_context = LegionTaskContext{task, variant_kind, *regions};
  const auto context       = legate::TaskContext{&legion_task_context};
  auto offset              = context.scalar(0).value<std::size_t>();
  auto size                = context.scalar(1).value<std::size_t>();

  const auto& future = task->futures[0];
  size               = std::min(size, future.get_untyped_size() - offset);

  auto mem_kind   = find_memory_kind_for_executing_processor();
  const auto* ptr = static_cast<const std::int8_t*>(future.get_buffer(mem_kind)) + offset;

  const Legion::UntypedDeferredValue return_value{size, mem_kind};
  const AccessorWO<std::int8_t, 1> acc{return_value, size, false};
  std::memcpy(acc.ptr(0), ptr, size);

  // Legion postamble
  return_value.finalize(legion_context);
}

template <VariantCode variant_id>
void register_extract_scalar_variant(const TaskInfo::RuntimeAddVariantKey& key,
                                     Library* core_lib,
                                     const std::unique_ptr<TaskInfo>& task_info,
                                     const VariantOptions& variant_options)
{
  // TODO(wonchanl): We could support Legion & Realm calling convensions so we don't pass nullptr
  // here. Should also remove the corresponding workaround function in TaskInfo!
  task_info->add_variant_(key,
                          legate::Library{core_lib},
                          variant_id,
                          &variant_options,
                          Legion::CodeDescriptor{extract_scalar_task<variant_id>});
}

// This task is launched for the side effect of having bloated instances created for the task and
// intended to do nothing otherwise.
class PrefetchBloatedInstances : public LegionTask<PrefetchBloatedInstances> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{CoreTask::PREFETCH_BLOATED_INSTANCES};

  static void cpu_variant(const Legion::Task* /*task*/,
                          const std::vector<Legion::PhysicalRegion>& /*regions*/,
                          Legion::Context /*context*/,
                          Legion::Runtime* /*runtime*/)
  {
  }
#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
  static void omp_variant(const Legion::Task* /*task*/,
                          const std::vector<Legion::PhysicalRegion>& /*regions*/,
                          Legion::Context /*context*/,
                          Legion::Runtime* /*runtime*/)
  {
  }
#endif
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  static void gpu_variant(const Legion::Task* /*task*/,
                          const std::vector<Legion::PhysicalRegion>& /*regions*/,
                          Legion::Context /*context*/,
                          Legion::Runtime* /*runtime*/)
  {
  }
#endif
};

}  // namespace

void register_legate_core_tasks(Library* core_lib)
{
  constexpr auto key     = TaskInfo::RuntimeAddVariantKey{};
  auto task_info         = std::make_unique<TaskInfo>("core::extract_scalar");
  constexpr auto options = VariantOptions{}.with_has_allocations(true);

  register_extract_scalar_variant<VariantCode::CPU>(key, core_lib, task_info, options);
  if (LEGATE_DEFINED(LEGATE_USE_CUDA)) {
    register_extract_scalar_variant<VariantCode::GPU>(
      key, core_lib, task_info, VariantOptions{options}.with_elide_device_ctx_sync(true));
  }
  if (LEGATE_DEFINED(LEGATE_USE_OPENMP)) {
    register_extract_scalar_variant<VariantCode::OMP>(key, core_lib, task_info, options);
  }
  core_lib->register_task(LocalTaskID{CoreTask::EXTRACT_SCALAR}, std::move(task_info));
  PrefetchBloatedInstances::register_variants(legate::Library{core_lib});

  register_array_tasks(core_lib);
  register_partitioning_tasks(core_lib);
  comm::register_tasks(core_lib);
  legate::experimental::io::detail::register_tasks();
  OffloadTo::register_variants(legate::Library{core_lib});
}

namespace {

[[nodiscard]] constexpr GlobalRedopID builtin_redop_id(ReductionOpKind op, Type::Code type_code)
{
  return static_cast<GlobalRedopID>(
    LEGION_REDOP_BASE +
    // FIXME(wonchanl): It's beyond my comprehension why this issue hasn't been triggered by any of
    // our tests until now, cause these reduction op IDs haven't changed since the beginning. In the
    // long run, we should register these built-in operators ourselves in Legate, instead of relying
    // on an equation that is loosely shared by Legate and Legion.
    //
    // We need to special-case the logical-AND reduction for booleans, as it is registered as a
    // prod reduction on the Legion side...
    (traits::detail::to_underlying(
       (type_code == Type::Code::BOOL && op == ReductionOpKind::AND) ? ReductionOpKind::MUL : op) *
     LEGION_TYPE_TOTAL) +
    traits::detail::to_underlying(type_code));
}

#define RECORD(OP, TYPE_CODE)                           \
  PrimitiveType{TYPE_CODE}.record_reduction_operator(   \
    traits::detail::to_underlying(ReductionOpKind::OP), \
    builtin_redop_id(ReductionOpKind::OP, TYPE_CODE));

#define RECORD_INT(OP)           \
  RECORD(OP, Type::Code::BOOL)   \
  RECORD(OP, Type::Code::INT8)   \
  RECORD(OP, Type::Code::INT16)  \
  RECORD(OP, Type::Code::INT32)  \
  RECORD(OP, Type::Code::INT64)  \
  RECORD(OP, Type::Code::UINT8)  \
  RECORD(OP, Type::Code::UINT16) \
  RECORD(OP, Type::Code::UINT32) \
  RECORD(OP, Type::Code::UINT64)

#define RECORD_FLOAT(OP)          \
  RECORD(OP, Type::Code::FLOAT16) \
  RECORD(OP, Type::Code::FLOAT32) \
  RECORD(OP, Type::Code::FLOAT64)

#define RECORD_COMPLEX(OP) RECORD(OP, Type::Code::COMPLEX64)

#define RECORD_ALL(OP) \
  RECORD_INT(OP)       \
  RECORD_FLOAT(OP)     \
  RECORD_COMPLEX(OP)

void register_builtin_reduction_ops()
{
  RECORD_ALL(ADD)
  RECORD(ADD, Type::Code::COMPLEX128)
  RECORD_ALL(MUL)

  RECORD_INT(MAX)
  RECORD_FLOAT(MAX)

  RECORD_INT(MIN)
  RECORD_FLOAT(MIN)

  RECORD_INT(OR)
  RECORD_INT(AND)
  RECORD_INT(XOR)
}

#undef RECORD_ALL
#undef RECORD_COMPLEX
#undef RECORD_FLOAT
#undef RECORD_INT
#undef RECORD
#undef BUILTIN_REDOP_ID

}  // namespace

extern void register_exception_reduction_op(const Library* context);

/*static*/ void Runtime::initialize_core_library_callback_(
  Legion::Machine,  // NOLINT(performance-unnecessary-value-param)
  Legion::Runtime*,
  const std::set<Processor>&)
{
  ResourceConfig config;
  config.max_tasks       = CoreTask::MAX_TASK;
  config.max_dyn_tasks   = config.max_tasks - CoreTask::FIRST_DYNAMIC_TASK;
  config.max_projections = traits::detail::to_underlying(CoreProjectionOp::MAX_FUNCTOR);
  // We register one sharding functor for each new projection functor
  config.max_shardings     = traits::detail::to_underlying(CoreShardID::MAX_FUNCTOR);
  config.max_reduction_ops = traits::detail::to_underlying(CoreReductionOp::MAX_REDUCTION);

  auto* runtime = Runtime::get_runtime();

  auto* const core_lib =
    runtime->create_library(CORE_LIBRARY_NAME, config, mapping::detail::create_core_mapper(), {});
  // Order is deliberate. core_library_() must be set here, because the core mapper and mapper
  // manager expect to call get_runtime()->core_library().
  runtime->core_library_ = core_lib;
  runtime->mapper_manager_.emplace();

  register_legate_core_tasks(core_lib);

  register_builtin_reduction_ops();

  register_exception_reduction_op(core_lib);

  register_legate_core_sharding_functors(core_lib);
}

CUstream Runtime::get_cuda_stream() const
{
  if constexpr (LEGATE_DEFINED(LEGATE_USE_CUDA)) {
    // The header-file is includable without CUDA, but the actual symbols are not compiled
    // (leading to link errors down the line) if Realm was not compiled with CUDA support.
    return Realm::Cuda::get_task_cuda_stream();
  }
  return nullptr;
}

const cuda::detail::CUDADriverAPI* Runtime::get_cuda_driver_api() { return cu_driver_.get(); }

cuda::detail::CUDAModuleManager* Runtime::get_cuda_module_manager()
{
  // This needs to be initialized in a thread-safe manner, so we do it in
  // Runtime::initialize(). Hence, we want this to throw if it is accessed before then, because
  // we can't guarantee it's threadsafe (and don't want to wrap this in some kind of mutex...).
  return &cu_mod_manager_.value();  // NOLINT(bugprone-unchecked-optional-access)
}

const MapperManager& Runtime::get_mapper_manager_() const
{
  // We want this to throw in debug mode
  // NOLINTBEGIN(bugprone-unchecked-optional-access)
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    return mapper_manager_.value();
  }
  return *mapper_manager_;
  // NOLINTEND(bugprone-unchecked-optional-access)
}

Processor Runtime::get_executing_processor() const
{
  // Cannot use member legion_context_ here since we may be calling this function from within a
  // task, where the context will have changed.
  return legion_runtime_->get_executing_processor(Legion::Runtime::get_context());
}

Legion::MapperID Runtime::mapper_id() const { return get_mapper_manager_().mapper_id(); }

bool has_started() { return the_runtime.state() == RuntimeManager::State::INITIALIZED; }

bool has_finished() { return the_runtime.state() == RuntimeManager::State::FINALIZED; }

}  // namespace legate::detail
