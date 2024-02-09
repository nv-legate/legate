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

#include "core/runtime/detail/runtime.h"

#include "core/comm/comm.h"
#include "core/data/detail/array_tasks.h"
#include "core/data/detail/external_allocation.h"
#include "core/data/detail/logical_array.h"
#include "core/data/detail/logical_region_field.h"
#include "core/data/detail/logical_store.h"
#include "core/mapping/detail/core_mapper.h"
#include "core/mapping/detail/default_mapper.h"
#include "core/mapping/detail/instance_manager.h"
#include "core/mapping/detail/machine.h"
#include "core/mapping/detail/mapping.h"
#include "core/operation/detail/copy.h"
#include "core/operation/detail/fill.h"
#include "core/operation/detail/gather.h"
#include "core/operation/detail/reduce.h"
#include "core/operation/detail/scatter.h"
#include "core/operation/detail/scatter_gather.h"
#include "core/operation/detail/task.h"
#include "core/operation/detail/task_launcher.h"
#include "core/partitioning/detail/partitioner.h"
#include "core/runtime/detail/library.h"
#include "core/runtime/detail/shard.h"
#include "core/runtime/runtime.h"
#include "core/task/detail/task_context.h"
#include "core/utilities/detail/enumerate.h"
#include "core/utilities/detail/hash.h"
#include "core/utilities/detail/strtoll.h"
#include "core/utilities/detail/tuple.h"
#include "core/utilities/detail/type_traits.h"

#include "env_defaults.h"
#include "realm/cmdline.h"
#include "realm/network.h"

#include <cinttypes>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace legate::detail {

Logger& log_legate()
{
  static Logger log{"legate"};

  return log;
}

void show_progress(const Legion::Task* task, Legion::Context ctx, Legion::Runtime* runtime);

/*static*/ bool Config::show_progress_requested = false;

/*static*/ bool Config::use_empty_task = false;

/*static*/ bool Config::synchronize_stream_view = false;

/*static*/ bool Config::log_mapping_decisions = false;

/*static*/ bool Config::has_socket_mem = false;

/*static*/ uint64_t Config::max_field_reuse_size = 0;

/*static*/ bool Config::warmup_nccl = false;

/*static*/ bool Config::log_partitioning_decisions = false;

namespace {

// This is the unique string name for our library which can be used from both C++ and Python to
// generate IDs
constexpr const char* const CORE_LIBRARY_NAME = "legate.core";
constexpr const char* const TOPLEVEL_NAME     = "Legate Core Toplevel Task";

}  // namespace

Runtime::Runtime()
  : legion_runtime_{Legion::Runtime::get_runtime()},
    field_reuse_freq_{
      extract_env("LEGATE_FIELD_REUSE_FREQ", FIELD_REUSE_FREQ_DEFAULT, FIELD_REUSE_FREQ_TEST)},
    force_consensus_match_{!!extract_env("LEGATE_CONSENSUS", CONSENSUS_DEFAULT, CONSENSUS_TEST)}
{
}

Library* Runtime::create_library(const std::string& library_name,
                                 const ResourceConfig& config,
                                 std::unique_ptr<mapping::Mapper> mapper,
                                 bool in_callback)
{
  if (libraries_.find(library_name) != libraries_.end()) {
    throw std::invalid_argument{"Library " + library_name + " already exists"};
  }

  log_legate().debug("Library %s is created", library_name.c_str());
  if (nullptr == mapper) {
    mapper = std::make_unique<mapping::detail::DefaultMapper>();
  }
  auto library             = std::unique_ptr<Library>{new Library{library_name, config}};
  auto ptr                 = library.get();
  libraries_[library_name] = std::move(library);
  ptr->register_mapper(std::move(mapper), in_callback);
  return ptr;
}

Library* Runtime::find_library(const std::string& library_name, bool can_fail /*=false*/) const
{
  const auto finder = libraries_.find(library_name);

  if (libraries_.end() == finder) {
    if (!can_fail) {
      throw std::out_of_range{"Library " + library_name + " does not exist"};
    }
    return {};
  }
  return finder->second.get();
}

Library* Runtime::find_or_create_library(const std::string& library_name,
                                         const ResourceConfig& config,
                                         std::unique_ptr<mapping::Mapper> mapper,
                                         bool* created,
                                         bool in_callback)
{
  auto result = find_library(library_name, true /*can_fail*/);

  if (result) {
    if (created) {
      *created = false;
    }
    return result;
  }
  result = create_library(library_name, config, std::move(mapper), in_callback);
  if (created != nullptr) {
    *created = true;
  }
  return result;
}

void Runtime::record_reduction_operator(uint32_t type_uid, int32_t op_kind, int32_t legion_op_id)
{
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    log_legate().debug("Record reduction op (type_uid: %d, op_kind: %d, legion_op_id: %d)",
                       type_uid,
                       op_kind,
                       legion_op_id);
  }
  auto key    = std::make_pair(type_uid, op_kind);
  auto finder = reduction_ops_.find(key);
  if (finder != reduction_ops_.end()) {
    std::stringstream ss;

    ss << "Reduction op " << op_kind << " already exists for type " << type_uid;
    throw std::invalid_argument{std::move(ss).str()};
  }
  reduction_ops_[key] = legion_op_id;
}

int32_t Runtime::find_reduction_operator(uint32_t type_uid, int32_t op_kind) const
{
  auto finder = reduction_ops_.find({type_uid, op_kind});
  if (reduction_ops_.end() == finder) {
    if (LegateDefined(LEGATE_USE_DEBUG)) {
      log_legate().debug("Can't find reduction op (type_uid: %d, op_kind: %d)", type_uid, op_kind);
    }
    std::stringstream ss;

    ss << "Reduction op " << op_kind << " does not exist for type " << type_uid;
    throw std::invalid_argument{std::move(ss).str()};
  }
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    log_legate().debug(
      "Found reduction op %d (type_uid: %d, op_kind: %d)", finder->second, type_uid, op_kind);
  }
  return finder->second;
}

void Runtime::initialize(Legion::Context legion_context)
{
  if (initialized_) {
    throw std::runtime_error{"Legate runtime has already been initialized"};
  }
  initialized_    = true;
  legion_context_ = legion_context;
  core_library_   = find_library(CORE_LIBRARY_NAME, false /*can_fail*/);
  // TODO(jfaibussowit): Use smart pointers for these
  communicator_manager_ = new CommunicatorManager{};
  partition_manager_    = new PartitionManager{this};
  machine_manager_      = new MachineManager{};
  provenance_manager_   = new ProvenanceManager{};
  Config::has_socket_mem =
    get_tunable<bool>(core_library_->get_mapper_id(), LEGATE_CORE_TUNABLE_HAS_SOCKET_MEM);
  Config::max_field_reuse_size = get_tunable<decltype(Config::max_field_reuse_size)>(
    core_library_->get_mapper_id(), LEGATE_CORE_TUNABLE_FIELD_REUSE_SIZE);
  initialize_toplevel_machine();
  comm::register_builtin_communicator_factories(core_library_);
}

mapping::detail::Machine Runtime::slice_machine_for_task(const Library* library, int64_t task_id)
{
  auto* task_info = library->find_task(task_id);
  auto& machine   = machine_manager_->get_machine();
  std::vector<mapping::TaskTarget> task_targets;

  for (const auto& t : machine.valid_targets()) {
    if (task_info->has_variant(mapping::detail::to_variant_code(t))) {
      task_targets.push_back(t);
    }
  }
  auto sliced = machine.only(task_targets);

  if (sliced.empty()) {
    std::stringstream ss;

    ss << "Task " << task_id << " (" << task_info->name() << ") of library "
       << library->get_library_name() << " does not have any valid variant for "
       << "the current machine configuration.";
    throw std::invalid_argument{std::move(ss).str()};
  }
  return sliced;
}

// This function should be moved to the library context
InternalSharedPtr<AutoTask> Runtime::create_task(const Library* library, int64_t task_id)
{
  auto machine = slice_machine_for_task(library, task_id);
  auto task = make_internal_shared<AutoTask>(library, task_id, current_op_id(), std::move(machine));
  increment_op_id();
  return task;
}

InternalSharedPtr<ManualTask> Runtime::create_task(const Library* library,
                                                   int64_t task_id,
                                                   const Domain& launch_domain)
{
  if (launch_domain.empty()) {
    throw std::invalid_argument{"Launch domain must not be empty"};
  }
  auto machine = slice_machine_for_task(library, task_id);
  auto task    = make_internal_shared<ManualTask>(
    library, task_id, launch_domain, current_op_id(), std::move(machine));
  increment_op_id();
  return task;
}

void Runtime::issue_copy(InternalSharedPtr<LogicalStore> target,
                         InternalSharedPtr<LogicalStore> source,
                         std::optional<int32_t> redop)
{
  auto machine = machine_manager_->get_machine();
  submit(make_internal_shared<Copy>(
    std::move(target), std::move(source), current_op_id(), std::move(machine), redop));
  increment_op_id();
}

void Runtime::issue_gather(InternalSharedPtr<LogicalStore> target,
                           InternalSharedPtr<LogicalStore> source,
                           InternalSharedPtr<LogicalStore> source_indirect,
                           std::optional<int32_t> redop)
{
  auto machine = machine_manager_->get_machine();
  submit(make_internal_shared<Gather>(std::move(target),
                                      std::move(source),
                                      std::move(source_indirect),
                                      current_op_id(),
                                      std::move(machine),
                                      redop));
  increment_op_id();
}

void Runtime::issue_scatter(InternalSharedPtr<LogicalStore> target,
                            InternalSharedPtr<LogicalStore> target_indirect,
                            InternalSharedPtr<LogicalStore> source,
                            std::optional<int32_t> redop)
{
  auto machine = machine_manager_->get_machine();
  submit(make_internal_shared<Scatter>(std::move(target),
                                       std::move(target_indirect),
                                       std::move(source),
                                       current_op_id(),
                                       std::move(machine),
                                       redop));
  increment_op_id();
}

void Runtime::issue_scatter_gather(InternalSharedPtr<LogicalStore> target,
                                   InternalSharedPtr<LogicalStore> target_indirect,
                                   InternalSharedPtr<LogicalStore> source,
                                   InternalSharedPtr<LogicalStore> source_indirect,
                                   std::optional<int32_t> redop)
{
  auto machine = machine_manager_->get_machine();
  submit(make_internal_shared<ScatterGather>(std::move(target),
                                             std::move(target_indirect),
                                             std::move(source),
                                             std::move(source_indirect),
                                             current_op_id(),
                                             std::move(machine),
                                             redop));
  increment_op_id();
}

void Runtime::issue_fill(const InternalSharedPtr<LogicalArray>& lhs,
                         InternalSharedPtr<LogicalStore> value)
{
  if (lhs->kind() != ArrayKind::BASE) {
    throw std::runtime_error{"Fills on list or struct arrays are not supported yet"};
  }

  if (value->type()->code == Type::Code::NIL) {
    if (!lhs->nullable()) {
      throw std::invalid_argument{"Non-nullable arrays cannot be filled with null"};
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
    throw std::runtime_error{"Fills on list or struct arrays are not supported yet"};
  }

  if (value.type()->code == Type::Code::NIL) {
    if (!lhs->nullable()) {
      throw std::invalid_argument{"Non-nullable arrays cannot be filled with null"};
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
    throw std::invalid_argument{"Fill lhs must be a normal store"};
  }
  if (!value->has_scalar_storage()) {
    throw std::invalid_argument{"Fill value should be a Future-back store"};
  }

  submit(make_internal_shared<Fill>(
    std::move(lhs), std::move(value), current_op_id(), machine_manager_->get_machine()));
  increment_op_id();
}

void Runtime::issue_fill(InternalSharedPtr<LogicalStore> lhs, Scalar value)
{
  if (lhs->unbound()) {
    throw std::invalid_argument{"Fill lhs must be a normal store"};
  }

  submit(make_internal_shared<Fill>(
    std::move(lhs), std::move(value), current_op_id(), machine_manager_->get_machine()));
  increment_op_id();
}

void Runtime::tree_reduce(const Library* library,
                          int64_t task_id,
                          InternalSharedPtr<LogicalStore> store,
                          InternalSharedPtr<LogicalStore> out_store,
                          int32_t radix)
{
  if (store->dim() != 1) {
    throw std::runtime_error{"Multi-dimensional stores are not supported"};
  }

  auto machine = machine_manager_->get_machine();
  submit(make_internal_shared<Reduce>(library,
                                      std::move(store),
                                      std::move(out_store),
                                      task_id,
                                      current_op_id(),
                                      radix,
                                      std::move(machine)));
  increment_op_id();
}

void Runtime::flush_scheduling_window()
{
  if (operations_.empty()) {
    return;
  }

  std::vector<InternalSharedPtr<Operation>> to_schedule;
  to_schedule.swap(operations_);
  schedule(to_schedule);
}

void Runtime::submit(InternalSharedPtr<Operation> op)
{
  op->validate();
  auto& submitted = operations_.emplace_back(std::move(op));
  if (submitted->always_flush() || operations_.size() >= window_size_) {
    flush_scheduling_window();
  }
}

void Runtime::schedule(const std::vector<InternalSharedPtr<Operation>>& operations)
{
  std::vector<Operation*> op_pointers{};

  op_pointers.reserve(operations.size());
  for (auto& op : operations) {
    op_pointers.push_back(op.get());
  }

  Partitioner partitioner{std::move(op_pointers)};
  auto strategy = partitioner.partition_stores();

  for (auto& op : operations) {
    op->launch(strategy.get());
  }
}

InternalSharedPtr<LogicalArray> Runtime::create_array(InternalSharedPtr<Type> type,
                                                      uint32_t dim,
                                                      bool nullable)
{
  // TODO(wonchanl): We should be able to control colocation of fields for struct types,
  // instead of special-casing rect types here
  if (Type::Code::STRUCT == type->code && !is_rect_type(type)) {
    return create_struct_array(std::move(type), dim, nullable);
  }
  if (type->variable_size()) {
    if (dim != 1) {
      throw std::invalid_argument{"List/string arrays can only be 1D"};
    }

    auto elem_type =
      Type::Code::STRING == type->code ? int8() : type->as_list_type().element_type();
    auto descriptor = create_base_array(rect_type(1), dim, nullable);
    auto vardata    = create_array(std::move(elem_type), 1, false);

    return make_internal_shared<ListLogicalArray>(
      std::move(type), std::move(descriptor), std::move(vardata));
  }
  return create_base_array(std::move(type), dim, nullable);
}

InternalSharedPtr<LogicalArray> Runtime::create_array(const InternalSharedPtr<Shape>& shape,
                                                      InternalSharedPtr<Type> type,
                                                      bool nullable,
                                                      bool optimize_scalar)
{
  // TODO(wonchanl): We should be able to control colocation of fields for struct types,
  // instead of special-casing rect types here
  if (Type::Code::STRUCT == type->code && !is_rect_type(type)) {
    return create_struct_array(shape, std::move(type), nullable, optimize_scalar);
  }

  if (type->variable_size()) {
    if (shape->ndim() != 1) {
      throw std::invalid_argument{"List/string arrays can only be 1D"};
    }

    auto elem_type =
      Type::Code::STRING == type->code ? int8() : type->as_list_type().element_type();
    auto descriptor = create_base_array(shape, rect_type(1), nullable, optimize_scalar);
    auto vardata    = create_array(std::move(elem_type), 1, false);

    return make_internal_shared<ListLogicalArray>(
      std::move(type), std::move(descriptor), std::move(vardata));
  }
  return create_base_array(shape, std::move(type), nullable, optimize_scalar);
}

InternalSharedPtr<LogicalArray> Runtime::create_array_like(
  const InternalSharedPtr<LogicalArray>& array, InternalSharedPtr<Type> type)
{
  if (Type::Code::STRUCT == type->code || type->variable_size()) {
    throw std::runtime_error{
      "create_array_like doesn't support variable size types or struct types"};
  }
  if (array->unbound()) {
    return create_array(std::move(type), array->dim(), array->nullable());
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
    throw std::invalid_argument{"Expected a list type but got " + type->to_string()};
  }
  if (descriptor->unbound() || vardata->unbound()) {
    throw std::invalid_argument("Sub-arrays should not be unbound");
  }
  if (descriptor->dim() != 1 || vardata->dim() != 1) {
    throw std::invalid_argument("Sub-arrays should be 1D");
  }
  if (!is_rect_type(descriptor->type(), 1)) {
    throw std::invalid_argument{"Descriptor array does not have a 1D rect type"};
  }
  // If this doesn't hold, something bad happened (and will happen below)
  LegateCheck(!descriptor->nested());
  if (vardata->nullable()) {
    throw std::invalid_argument{"Vardata should not be nullable"};
  }

  auto elem_type = Type::Code::STRING == type->code ? int8() : type->as_list_type().element_type();
  if (*vardata->type() != *elem_type) {
    throw std::invalid_argument{"Expected a vardata array of type " + elem_type->to_string() +
                                " but got " + vardata->type()->to_string()};
  }

  return make_internal_shared<ListLogicalArray>(
    std::move(type), legate::static_pointer_cast<BaseLogicalArray>(descriptor), std::move(vardata));
}

InternalSharedPtr<StructLogicalArray> Runtime::create_struct_array(InternalSharedPtr<Type> type,
                                                                   uint32_t dim,
                                                                   bool nullable)
{
  std::vector<InternalSharedPtr<LogicalArray>> fields;
  const auto& st_type = type->as_struct_type();
  auto null_mask      = nullable ? create_store(bool_(), dim) : nullptr;

  fields.reserve(st_type.field_types().size());
  for (auto& field_type : st_type.field_types()) {
    fields.emplace_back(create_array(field_type, dim, false));
  }
  return make_internal_shared<StructLogicalArray>(
    std::move(type), std::move(null_mask), std::move(fields));
}

InternalSharedPtr<StructLogicalArray> Runtime::create_struct_array(
  const InternalSharedPtr<Shape>& shape,
  InternalSharedPtr<Type> type,
  bool nullable,
  bool optimize_scalar)
{
  std::vector<InternalSharedPtr<LogicalArray>> fields;
  const auto& st_type = type->as_struct_type();
  auto null_mask      = nullable ? create_store(shape, bool_(), optimize_scalar) : nullptr;

  fields.reserve(st_type.field_types().size());
  for (auto& field_type : st_type.field_types()) {
    fields.emplace_back(create_array(shape, field_type, false, optimize_scalar));
  }
  return make_internal_shared<StructLogicalArray>(
    std::move(type), std::move(null_mask), std::move(fields));
}

InternalSharedPtr<BaseLogicalArray> Runtime::create_base_array(InternalSharedPtr<Type> type,
                                                               uint32_t dim,
                                                               bool nullable)
{
  auto data      = create_store(std::move(type), dim);
  auto null_mask = nullable ? create_store(bool_(), dim) : nullptr;
  return make_internal_shared<BaseLogicalArray>(std::move(data), std::move(null_mask));
}

InternalSharedPtr<BaseLogicalArray> Runtime::create_base_array(
  const InternalSharedPtr<Shape>& shape,
  InternalSharedPtr<Type> type,
  bool nullable,
  bool optimize_scalar)
{
  auto null_mask = nullable ? create_store(shape, bool_(), optimize_scalar) : nullptr;
  auto data      = create_store(shape, std::move(type), optimize_scalar);
  return make_internal_shared<BaseLogicalArray>(std::move(data), std::move(null_mask));
}

InternalSharedPtr<LogicalStore> Runtime::create_store(InternalSharedPtr<Type> type, uint32_t dim)
{
  check_dimensionality(dim);
  auto storage = make_internal_shared<detail::Storage>(dim, std::move(type));
  return make_internal_shared<LogicalStore>(std::move(storage));
}

InternalSharedPtr<LogicalStore> Runtime::create_store(const InternalSharedPtr<Shape>& shape,
                                                      InternalSharedPtr<Type> type,
                                                      bool optimize_scalar /*=false*/)
{
  check_dimensionality(shape->ndim());
  auto storage = make_internal_shared<detail::Storage>(shape, std::move(type), optimize_scalar);
  return make_internal_shared<LogicalStore>(std::move(storage));
}

InternalSharedPtr<LogicalStore> Runtime::create_store(const Scalar& scalar,
                                                      const InternalSharedPtr<Shape>& shape)
{
  if (shape->volume() != 1) {
    throw std::invalid_argument{"Scalar stores must have a shape of volume 1"};
  }
  auto future  = Legion::Future::from_untyped_pointer(scalar.data(), scalar.size());
  auto storage = make_internal_shared<detail::Storage>(shape, scalar.type(), future);
  return make_internal_shared<detail::LogicalStore>(std::move(storage));
}

InternalSharedPtr<LogicalStore> Runtime::create_store(
  const InternalSharedPtr<Shape>& shape,
  InternalSharedPtr<Type> type,
  InternalSharedPtr<ExternalAllocation> allocation,
  const mapping::detail::DimOrdering* ordering)
{
  if (type->variable_size()) {
    throw std::invalid_argument{
      "stores created by attaching to a buffer must have fixed-size type"};
  }
  LegateCheck(allocation->ptr());
  if (allocation->size() < shape->volume() * type->size()) {
    throw std::invalid_argument{"External allocation of size " +
                                std::to_string(allocation->size()) +
                                " is not big enough "
                                "for a store of shape " +
                                shape->extents().to_string() + " and type " + type->to_string()};
  }

  InternalSharedPtr<LogicalStore> store =
    create_store(shape, std::move(type), false /*optimize_scalar*/);
  const InternalSharedPtr<LogicalRegionField> rf = store->get_region_field();

  Legion::AttachLauncher launcher{LEGION_EXTERNAL_INSTANCE,
                                  rf->region(),
                                  rf->region(),
                                  false /*restricted*/,
                                  !allocation->read_only() /*mapped*/};
  launcher.collective = true;  // each shard will attach a full local copy of the entire buffer
  launcher.provenance = provenance_manager()->get_provenance();
  launcher.constraints.ordering_constraint.ordering.clear();
  ordering->populate_dimension_ordering(store->dim(),
                                        launcher.constraints.ordering_constraint.ordering);
  launcher.constraints.ordering_constraint.ordering.push_back(DIM_F);
  launcher.constraints.field_constraint =
    Legion::FieldConstraint{std::vector<Legion::FieldID>{rf->field_id()}, false, false};
  launcher.privilege_fields.insert(rf->field_id());
  launcher.external_resource = allocation->resource();

  auto pr = legion_runtime_->attach_external_resource(legion_context_, launcher);
  // no need to wait on the returned PhysicalRegion, since we're not inline-mapping
  // but we can keep it around and remap it later if the user asks
  rf->attach(std::move(pr), std::move(allocation));

  return store;
}

Runtime::IndexAttachResult Runtime::create_store(
  const InternalSharedPtr<Shape>& shape,
  const tuple<uint64_t>& tile_shape,
  InternalSharedPtr<Type> type,
  const std::vector<std::pair<legate::ExternalAllocation, tuple<uint64_t>>>& allocations,
  const mapping::detail::DimOrdering* ordering)
{
  if (type->variable_size()) {
    throw std::invalid_argument{
      "stores created by attaching to a buffer must have fixed-size type"};
  }

  auto type_size = type->size();
  auto store     = create_store(shape, std::move(type), false /*optimize_scalar*/);
  auto partition = partition_store_by_tiling(store, tile_shape);

  auto rf = store->get_region_field();

  Legion::IndexAttachLauncher launcher{
    LEGION_EXTERNAL_INSTANCE, rf->region(), false /*restricted*/};

  std::vector<InternalSharedPtr<ExternalAllocation>> allocs;
  std::unordered_set<uint64_t> visited;
  const hasher<tuple<uint64_t>> hash_color{};
  visited.reserve(allocations.size());
  allocs.reserve(allocations.size());
  for (auto&& [idx, spec] : enumerate(allocations)) {
    auto&& [allocation, color] = spec;
    const auto color_hash      = hash_color(color);
    if (visited.find(color_hash) != visited.end()) {
      // If we're here, this color might have been seen in one of the previous iterations
      for (int64_t k = 0; k < idx; ++k) {
        if (allocations[k].second == color) {
          throw std::invalid_argument{"Mulitple external allocations are found for color " +
                                      color.to_string()};
        }
      }
      // If we're here, then we've just seen a fairly rare hash collision
    }
    visited.insert(color_hash);
    auto& alloc        = allocs.emplace_back(allocation.impl());
    auto substore      = partition->get_child_store(color);
    auto required_size = substore->volume() * type_size;

    LegateAssert(alloc->ptr());

    if (!alloc->read_only()) {
      throw std::invalid_argument{"External allocations must be read-only"};
    }

    if (required_size > alloc->size()) {
      throw std::invalid_argument{"Sub-store for color " + color.to_string() +
                                  " requires the allocation "
                                  "to be at least " +
                                  std::to_string(required_size) +
                                  " bytes, but the allocation "
                                  "is only " +
                                  std::to_string(alloc->size()) + " bytes"};
    }

    launcher.add_external_resource(substore->get_region_field()->region(), alloc->resource());
  }
  launcher.provenance = provenance_manager()->get_provenance();
  launcher.constraints.ordering_constraint.ordering.clear();
  ordering->populate_dimension_ordering(store->dim(),
                                        launcher.constraints.ordering_constraint.ordering);
  launcher.constraints.ordering_constraint.ordering.push_back(DIM_F);
  launcher.constraints.field_constraint =
    Legion::FieldConstraint{std::vector<Legion::FieldID>{rf->field_id()}, false, false};
  launcher.privilege_fields.insert(rf->field_id());

  auto external_resources = legion_runtime_->attach_external_resources(legion_context_, launcher);
  rf->attach(external_resources, std::move(allocs));

  return {std::move(store), std::move(partition)};
}

void Runtime::check_dimensionality(uint32_t dim)
{
  if (dim > LEGATE_MAX_DIM) {
    throw std::out_of_range{"The maximum number of dimensions is " +
                            std::to_string(LEGION_MAX_DIM) + ", but a " + std::to_string(dim) +
                            "-D store is requested"};
  }
}

void Runtime::raise_pending_task_exception()
{
  auto exn = check_pending_task_exception();
  if (exn.has_value()) {
    throw std::move(exn).value();
  }
}

std::optional<TaskException> Runtime::check_pending_task_exception()
{
  // If there's already an outstanding exception from the previous scan, we just return that.
  if (!outstanding_exceptions_.empty()) {
    auto result = std::make_optional(std::move(outstanding_exceptions_.front()));

    outstanding_exceptions_.pop_front();
    return result;
  }

  // Otherwise, we unpack all pending exceptions and push them to the outstanding exception queue
  for (auto& pending_exception : pending_exceptions_) {
    auto returned_exception = pending_exception.get_result<ReturnedException>();
    auto result             = returned_exception.to_task_exception();

    if (result.has_value()) {
      outstanding_exceptions_.emplace_back(std::move(result).value());
    }
  }
  pending_exceptions_.clear();
  return outstanding_exceptions_.empty() ? std::nullopt : check_pending_task_exception();
}

void Runtime::record_pending_exception(const Legion::Future& pending_exception)
{
  pending_exceptions_.push_back(pending_exception);
  raise_pending_task_exception();
}

InternalSharedPtr<LogicalRegionField> Runtime::create_region_field(
  const InternalSharedPtr<Shape>& shape, uint32_t field_size)
{
  return find_or_create_field_manager(shape, field_size)->allocate_field();
}

InternalSharedPtr<LogicalRegionField> Runtime::import_region_field(
  const InternalSharedPtr<Shape>& shape,
  Legion::LogicalRegion region,
  Legion::FieldID field_id,
  uint32_t field_size)
{
  return find_or_create_field_manager(shape, field_size)->import_field(region, field_id);
}

Legion::PhysicalRegion Runtime::map_region_field(Legion::LogicalRegion region,
                                                 Legion::FieldID field_id)
{
  Legion::RegionRequirement req(region, LEGION_READ_WRITE, EXCLUSIVE, region);
  req.add_field(field_id);
  auto mapper_id = core_library_->get_mapper_id();
  // TODO(wonchanl): We need to pass the metadata about logical store
  Legion::InlineLauncher launcher{req, mapper_id};
  launcher.provenance       = provenance_manager()->get_provenance();
  Legion::PhysicalRegion pr = legion_runtime_->map_region(legion_context_, launcher);
  pr.wait_until_valid(true /*silence_warnings*/);
  return pr;
}

void Runtime::remap_physical_region(Legion::PhysicalRegion pr)
{
  legion_runtime_->remap_region(
    legion_context_, pr, provenance_manager()->get_provenance().c_str());
  pr.wait_until_valid(true /*silence_warnings*/);
}

void Runtime::unmap_physical_region(Legion::PhysicalRegion pr)
{
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    std::vector<Legion::FieldID> fields;
    pr.get_fields(fields);
    LegateCheck(fields.size() == 1);
  }
  legion_runtime_->unmap_region(legion_context_, std::move(pr));
}

Legion::Future Runtime::detach(const Legion::PhysicalRegion& physical_region,
                               bool flush,
                               bool unordered)
{
  LegateCheck(physical_region.exists() && !physical_region.is_mapped());
  return legion_runtime_->detach_external_resource(legion_context_,
                                                   physical_region,
                                                   flush,
                                                   unordered,
                                                   provenance_manager()->get_provenance().c_str());
}

Legion::Future Runtime::detach(const Legion::ExternalResources& external_resources,
                               bool flush,
                               bool unordered)
{
  LegateCheck(external_resources.exists());
  return legion_runtime_->detach_external_resources(legion_context_,
                                                    external_resources,
                                                    flush,
                                                    unordered,
                                                    provenance_manager()->get_provenance().c_str());
}

bool Runtime::consensus_match_required() const
{
  return force_consensus_match_ || Legion::Machine::get_machine().get_address_space_count() > 1;
}

void Runtime::progress_unordered_operations() const
{
  legion_runtime_->progress_unordered_operations(legion_context_);
}

RegionManager* Runtime::find_or_create_region_manager(const Legion::IndexSpace& index_space)
{
  auto finder = region_managers_.find(index_space);
  if (finder != region_managers_.end()) {
    return finder->second.get();
  }

  auto rgn_mgr                  = std::make_unique<RegionManager>(this, index_space);
  auto ptr                      = rgn_mgr.get();
  region_managers_[index_space] = std::move(rgn_mgr);
  return ptr;
}

FieldManager* Runtime::find_or_create_field_manager(const InternalSharedPtr<Shape>& shape,
                                                    uint32_t field_size)
{
  auto key    = FieldManagerKey(shape->index_space(), field_size);
  auto finder = field_managers_.find(key);
  if (finder != field_managers_.end()) {
    return finder->second.get();
  }

  auto fld_mgr         = consensus_match_required()
                           ? std::make_unique<ConsensusMatchingFieldManager>(this, shape, field_size)
                           : std::make_unique<FieldManager>(this, shape, field_size);
  auto ptr             = fld_mgr.get();
  field_managers_[key] = std::move(fld_mgr);
  return ptr;
}

const Legion::IndexSpace& Runtime::find_or_create_index_space(const tuple<uint64_t>& extents)
{
  if (extents.size() > LEGATE_MAX_DIM) {
    throw std::out_of_range("Legate is configured with the maximum number of dimensions set to " +
                            std::to_string(LEGATE_MAX_DIM) + ", but got a " +
                            std::to_string(extents.size()) + "-D shape");
  }

  return find_or_create_index_space(to_domain(extents));
}

const Legion::IndexSpace& Runtime::find_or_create_index_space(const Domain& domain)
{
  LegateCheck(nullptr != legion_context_);
  auto finder = cached_index_spaces_.find(domain);
  if (finder != cached_index_spaces_.end()) {
    return finder->second;
  }

  auto [it, _] = cached_index_spaces_.emplace(
    domain, legion_runtime_->create_index_space(legion_context_, domain));
  return it->second;
}

Legion::IndexPartition Runtime::create_restricted_partition(
  const Legion::IndexSpace& index_space,
  const Legion::IndexSpace& color_space,
  Legion::PartitionKind kind,
  const Legion::DomainTransform& transform,
  const Domain& extent)
{
  return legion_runtime_->create_partition_by_restriction(
    legion_context_, index_space, color_space, transform, extent, kind);
}

Legion::IndexPartition Runtime::create_weighted_partition(const Legion::IndexSpace& index_space,
                                                          const Legion::IndexSpace& color_space,
                                                          const Legion::FutureMap& weights)
{
  return legion_runtime_->create_partition_by_weights(
    legion_context_, index_space, weights, color_space);
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
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Create image partition {index_space: " << index_space
                         << ", func_partition: " << func_partition
                         << ", func_field_id: " << func_field_id << ", is_range: " << is_range
                         << "}";
  }

  BufferBuilder buffer;
  machine.pack(buffer);
  buffer.pack<uint32_t>(get_sharding(machine, 0));

  if (is_range) {
    return legion_runtime_->create_partition_by_image_range(legion_context_,
                                                            index_space,
                                                            func_partition,
                                                            func_region,
                                                            func_field_id,
                                                            color_space,
                                                            LEGION_COMPUTE_KIND,
                                                            LEGION_AUTO_GENERATE_ID,
                                                            core_library_->get_mapper_id(),
                                                            0,
                                                            buffer.to_legion_buffer());
  }
  return legion_runtime_->create_partition_by_image(legion_context_,
                                                    index_space,
                                                    func_partition,
                                                    func_region,
                                                    func_field_id,
                                                    color_space,
                                                    LEGION_COMPUTE_KIND,
                                                    LEGION_AUTO_GENERATE_ID,
                                                    core_library_->get_mapper_id(),
                                                    0,
                                                    buffer.to_legion_buffer());
}

Legion::FieldSpace Runtime::create_field_space()
{
  LegateCheck(nullptr != legion_context_);
  return legion_runtime_->create_field_space(legion_context_);
}

Legion::LogicalRegion Runtime::create_region(const Legion::IndexSpace& index_space,
                                             const Legion::FieldSpace& field_space)
{
  LegateCheck(nullptr != legion_context_);
  return legion_runtime_->create_logical_region(legion_context_, index_space, field_space);
}

void Runtime::destroy_region(const Legion::LogicalRegion& logical_region, bool unordered)
{
  LegateCheck(nullptr != legion_context_);
  legion_runtime_->destroy_logical_region(legion_context_, logical_region, unordered);
}

Legion::LogicalPartition Runtime::create_logical_partition(
  const Legion::LogicalRegion& logical_region, const Legion::IndexPartition& index_partition)
{
  LegateCheck(nullptr != legion_context_);
  return legion_runtime_->get_logical_partition(legion_context_, logical_region, index_partition);
}

Legion::LogicalRegion Runtime::get_subregion(const Legion::LogicalPartition& partition,
                                             const Legion::DomainPoint& color)
{
  LegateCheck(nullptr != legion_context_);
  return legion_runtime_->get_logical_subregion_by_color(legion_context_, partition, color);
}

Legion::LogicalRegion Runtime::find_parent_region(const Legion::LogicalRegion& region)
{
  auto result = region;
  while (legion_runtime_->has_parent_logical_partition(legion_context_, result)) {
    auto partition = legion_runtime_->get_parent_logical_partition(legion_context_, result);
    result         = legion_runtime_->get_parent_logical_region(legion_context_, partition);
  }
  return result;
}

Legion::FieldID Runtime::allocate_field(const Legion::FieldSpace& field_space, size_t field_size)
{
  LegateCheck(nullptr != legion_context_);
  auto allocator = legion_runtime_->create_field_allocator(legion_context_, field_space);
  return allocator.allocate_field(field_size);
}

Legion::FieldID Runtime::allocate_field(const Legion::FieldSpace& field_space,
                                        Legion::FieldID field_id,
                                        size_t field_size)
{
  LegateCheck(nullptr != legion_context_);
  auto allocator = legion_runtime_->create_field_allocator(legion_context_, field_space);
  return allocator.allocate_field(field_size, field_id);
}

Domain Runtime::get_index_space_domain(const Legion::IndexSpace& index_space) const
{
  LegateCheck(nullptr != legion_context_);
  return legion_runtime_->get_index_space_domain(legion_context_, index_space);
}

namespace {

Legion::DomainPoint _delinearize_future_map(const DomainPoint& point,
                                            const Domain& domain,
                                            const Domain& range)
{
  LegateCheck(range.dim == 1);
  DomainPoint result;
  result.dim = 1;

  const int32_t ndim = domain.dim;
  int64_t idx        = point[0];
  for (int32_t dim = 1; dim < ndim; ++dim) {
    const int64_t extent = domain.rect_data[dim + ndim] - domain.rect_data[dim] + 1;
    idx                  = idx * extent + point[dim];
  }
  result[0] = idx;
  return result;
}

}  // namespace

Legion::FutureMap Runtime::delinearize_future_map(const Legion::FutureMap& future_map,
                                                  const Legion::IndexSpace& new_domain) const
{
  return legion_runtime_->transform_future_map(
    legion_context_, future_map, new_domain, _delinearize_future_map);
}

std::pair<Legion::PhaseBarrier, Legion::PhaseBarrier> Runtime::create_barriers(size_t num_tasks)
{
  auto arrival_barrier = legion_runtime_->create_phase_barrier(legion_context_, num_tasks);
  auto wait_barrier    = legion_runtime_->advance_phase_barrier(legion_context_, arrival_barrier);
  return {arrival_barrier, wait_barrier};
}

void Runtime::destroy_barrier(Legion::PhaseBarrier barrier)
{
  legion_runtime_->destroy_phase_barrier(legion_context_, barrier);
}

Legion::Future Runtime::get_tunable(Legion::MapperID mapper_id, int64_t tunable_id, size_t size)
{
  const Legion::TunableLauncher launcher{
    static_cast<Legion::TunableID>(tunable_id), mapper_id, 0, size};
  return legion_runtime_->select_tunable_value(legion_context_, launcher);
}

Legion::Future Runtime::dispatch(Legion::TaskLauncher& launcher,
                                 std::vector<Legion::OutputRequirement>& output_requirements)
{
  LegateCheck(nullptr != legion_context_);
  return legion_runtime_->execute_task(legion_context_, launcher, &output_requirements);
}

Legion::FutureMap Runtime::dispatch(Legion::IndexTaskLauncher& launcher,
                                    std::vector<Legion::OutputRequirement>& output_requirements)
{
  LegateCheck(nullptr != legion_context_);
  return legion_runtime_->execute_index_space(legion_context_, launcher, &output_requirements);
}

void Runtime::dispatch(Legion::CopyLauncher& launcher)
{
  LegateCheck(nullptr != legion_context_);
  return legion_runtime_->issue_copy_operation(legion_context_, launcher);
}

void Runtime::dispatch(Legion::IndexCopyLauncher& launcher)
{
  LegateCheck(nullptr != legion_context_);
  return legion_runtime_->issue_copy_operation(legion_context_, launcher);
}

void Runtime::dispatch(Legion::FillLauncher& launcher)
{
  LegateCheck(nullptr != legion_context_);
  return legion_runtime_->fill_fields(legion_context_, launcher);
}

void Runtime::dispatch(Legion::IndexFillLauncher& launcher)
{
  LegateCheck(nullptr != legion_context_);
  return legion_runtime_->fill_fields(legion_context_, launcher);
}

Legion::Future Runtime::extract_scalar(const Legion::Future& result, uint32_t idx) const
{
  auto& machine    = get_machine();
  auto& provenance = provenance_manager()->get_provenance();
  auto variant     = mapping::detail::to_variant_code(machine.preferred_target);
  auto launcher =
    TaskLauncher{core_library_, machine, provenance, LEGATE_CORE_EXTRACT_SCALAR_TASK_ID, variant};

  launcher.add_future(result);
  launcher.add_scalar(Scalar(idx));
  return launcher.execute_single();
}

Legion::FutureMap Runtime::extract_scalar(const Legion::FutureMap& result,
                                          uint32_t idx,
                                          const Legion::Domain& launch_domain) const
{
  auto& machine    = get_machine();
  auto& provenance = provenance_manager()->get_provenance();
  auto variant     = mapping::detail::to_variant_code(machine.preferred_target);
  auto launcher =
    TaskLauncher{core_library_, machine, provenance, LEGATE_CORE_EXTRACT_SCALAR_TASK_ID, variant};

  launcher.add_future_map(result);
  launcher.add_scalar(Scalar(idx));
  return launcher.execute(launch_domain);
}

Legion::Future Runtime::reduce_future_map(const Legion::FutureMap& future_map,
                                          int32_t reduction_op,
                                          const Legion::Future& init_value) const
{
  return legion_runtime_->reduce_future_map(legion_context_,
                                            future_map,
                                            reduction_op,
                                            false /*deterministic*/,
                                            core_library_->get_mapper_id(),
                                            0 /*tag*/,
                                            nullptr /*provenance*/,
                                            init_value);
}

Legion::Future Runtime::reduce_exception_future_map(const Legion::FutureMap& future_map) const
{
  auto reduction_op = core_library_->get_reduction_op_id(LEGATE_CORE_JOIN_EXCEPTION_OP);
  return legion_runtime_->reduce_future_map(legion_context_,
                                            future_map,
                                            reduction_op,
                                            false /*deterministic*/,
                                            core_library_->get_mapper_id(),
                                            LEGATE_CORE_JOIN_EXCEPTION_TAG);
}

void Runtime::discard_field(const Legion::LogicalRegion& region, Legion::FieldID field_id)
{
  Legion::DiscardLauncher launcher{region, region};
  launcher.add_field(field_id);
  legion_runtime_->discard_fields(legion_context_, launcher);
}

void Runtime::issue_execution_fence(bool block /*=false*/)
{
  flush_scheduling_window();
  // FIXME: This needs to be a Legate operation
  auto future = legion_runtime_->issue_execution_fence(legion_context_);
  if (block) {
    future.wait();
  }
}

void Runtime::initialize_toplevel_machine()
{
  auto mapper_id    = core_library_->get_mapper_id();
  auto num_nodes    = get_tunable<uint32_t>(mapper_id, LEGATE_CORE_TUNABLE_NUM_NODES);
  auto num_gpus     = get_tunable<uint32_t>(mapper_id, LEGATE_CORE_TUNABLE_TOTAL_GPUS);
  auto num_omps     = get_tunable<uint32_t>(mapper_id, LEGATE_CORE_TUNABLE_TOTAL_OMPS);
  auto num_cpus     = get_tunable<uint32_t>(mapper_id, LEGATE_CORE_TUNABLE_TOTAL_CPUS);
  auto create_range = [&num_nodes](uint32_t num_procs) {
    auto per_node_count = static_cast<uint32_t>(num_procs / num_nodes);
    return mapping::ProcessorRange{0, num_procs, per_node_count};
  };

  mapping::detail::Machine machine{{{mapping::TaskTarget::GPU, create_range(num_gpus)},
                                    {mapping::TaskTarget::OMP, create_range(num_omps)},
                                    {mapping::TaskTarget::CPU, create_range(num_cpus)}}};

  LegateAssert(machine_manager_ != nullptr);
  machine_manager_->push_machine(std::move(machine));
}

const mapping::detail::Machine& Runtime::get_machine() const
{
  LegateAssert(machine_manager_ != nullptr);
  return machine_manager_->get_machine();
}

Legion::ProjectionID Runtime::get_affine_projection(uint32_t src_ndim,
                                                    const proj::SymbolicPoint& point)
{
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Query affine projection {src_ndim: " << src_ndim
                         << ", point: " << point << "}";
  }

  if (proj::is_identity(src_ndim, point)) {
    if (LegateDefined(LEGATE_USE_DEBUG)) {
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

  auto proj_id = core_library_->get_projection_id(next_projection_id_++);

  register_affine_projection_functor(src_ndim, point, proj_id);
  affine_projections_[key] = proj_id;

  if (LegateDefined(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Register affine projection " << proj_id << " {src_ndim: " << src_ndim
                         << ", point: " << point << "}";
  }

  return proj_id;
}

Legion::ProjectionID Runtime::get_delinearizing_projection(const tuple<uint64_t>& color_shape)
{
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Query delinearizing projection {color_shape: "
                         << color_shape.to_string() << "}";
  }

  auto finder = delinearizing_projections_.find(color_shape);
  if (delinearizing_projections_.end() != finder) {
    return finder->second;
  }

  auto proj_id = core_library_->get_projection_id(next_projection_id_++);

  register_delinearizing_projection_functor(color_shape, proj_id);
  delinearizing_projections_[color_shape] = proj_id;

  if (LegateDefined(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Register delinearizing projection " << proj_id
                         << "{color_shape: " << color_shape.to_string() << "}";
  }

  return proj_id;
}

Legion::ProjectionID Runtime::get_compound_projection(const tuple<uint64_t>& color_shape,
                                                      const proj::SymbolicPoint& point)
{
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Query compound projection {color_shape: " << color_shape.to_string()
                         << ", point: " << point << "}";
  }

  auto key    = CompoundProjectionDesc{color_shape, point};
  auto finder = compound_projections_.find(key);
  if (compound_projections_.end() != finder) {
    return finder->second;
  }

  auto proj_id = core_library_->get_projection_id(next_projection_id_++);

  register_compound_projection_functor(color_shape, point, proj_id);
  compound_projections_[key] = proj_id;

  if (LegateDefined(LEGATE_USE_DEBUG)) {
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

  if (LegateDefined(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Query sharding {proj_id: " << proj_id
                         << ", processor range: " << proc_range
                         << ", processor type: " << machine.preferred_target << "}";
  }

  auto finder = registered_shardings_.find(key);
  if (finder != registered_shardings_.end()) {
    if (LegateDefined(LEGATE_USE_DEBUG)) {
      log_legate().debug() << "Found sharding " << finder->second;
    }
    return finder->second;
  }

  auto sharding_id = core_library_->get_sharding_id(next_sharding_id_++);
  registered_shardings_.insert({std::move(key), sharding_id});

  if (LegateDefined(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Create sharding " << sharding_id;
  }

  create_sharding_functor_using_projection(sharding_id, proj_id, proc_range);

  return sharding_id;
}

/*static*/ int32_t Runtime::start(int32_t argc, char** argv)
{
  int32_t result = 0;

  if (!Legion::Runtime::has_runtime()) {
    Legion::Runtime::initialize(&argc, &argv, /*filter=*/false, /*parse=*/false);

    Legion::Runtime::perform_registration_callback(initialize_core_library_callback,
                                                   true /*global*/);

    handle_legate_args(argc, argv);

    result = Legion::Runtime::start(argc, argv, /*background=*/true);
    if (result != 0) {
      log_legate().error("Legion Runtime failed to start.");
      return result;
    }
  } else {
    Legion::Runtime::perform_registration_callback(initialize_core_library_callback,
                                                   true /*global*/);
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
    legion_context = legion_runtime->begin_implicit_task(LEGATE_CORE_TOPLEVEL_TASK_ID,
                                                         0 /*mapper id*/,
                                                         Processor::LOC_PROC,
                                                         TOPLEVEL_NAME,
                                                         true /*control replicable*/);
  }

  // We can now initialize the Legate runtime with the Legion context
  Runtime::get_runtime()->initialize(legion_context);
  return result;
}

int32_t Runtime::finish()
{
  if (!initialized()) {
    return 0;
  }
  destroy();
  // Mark that we are done excecuting the top-level task
  // After this call the context is no longer valid
  Legion::Runtime::get_runtime()->finish_implicit_task(std::exchange(legion_context_, nullptr));

  // The previous call is asynchronous so we still need to
  // wait for the shutdown of the runtime to complete
  return Legion::Runtime::wait_for_shutdown();
}

/*static*/ Runtime* Runtime::get_runtime()
{
  static auto runtime = std::make_unique<Runtime>();
  return runtime.get();
}

void Runtime::destroy()
{
  if (!initialized()) {
    return;
  }

  if (LegateDefined(LEGATE_USE_DEBUG)) {
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
  communicator_manager_->destroy();

  // Destroy all Legion handles used by Legate
  for (auto& [_, region_manager] : region_managers_) {
    region_manager->destroy(true /*unordered*/);
  }
  for (auto& [_, index_space] : cached_index_spaces_) {
    legion_runtime_->destroy_index_space(legion_context_, index_space, true /*unordered*/);
  }
  cached_index_spaces_.clear();

  // We're about to deallocate objects below, so let's block on all outstanding Legion operations
  issue_execution_fence(true);

  mapping::detail::InstanceManager::get_instance_manager()->destroy();
  mapping::detail::ReductionInstanceManager::get_instance_manager()->destroy();

  // Any STL containers holding Legion handles need to be cleared here, otherwise they cause
  // trouble when they get destroyed in the Legate runtime's destructor
  pending_exceptions_.clear();

  // We finally deallocate managers
  for (auto& [_, library] : libraries_) {
    library.reset();
  }
  libraries_.clear();
  for (auto& [_, region_manager] : region_managers_) {
    region_manager.reset();
  }
  region_managers_.clear();
  for (auto& [_, field_manager] : field_managers_) {
    field_manager.reset();
  }
  field_managers_.clear();

  delete std::exchange(communicator_manager_, nullptr);
  delete std::exchange(machine_manager_, nullptr);
  delete std::exchange(partition_manager_, nullptr);
  delete std::exchange(provenance_manager_, nullptr);
  initialized_ = false;
}

namespace {

template <LegateVariantCode variant_kind>
void extract_scalar_task(const void* args,
                         size_t arglen,
                         const void* /*userdata*/,
                         size_t /*userlen*/,
                         Legion::Processor p)
{
  // Legion preamble
  const Legion::Task* task;
  const std::vector<Legion::PhysicalRegion>* regions;
  Legion::Context legion_context;
  Legion::Runtime* runtime;
  Legion::Runtime::legion_task_preamble(args, arglen, p, task, regions, legion_context, runtime);

  legate::detail::show_progress(task, legion_context, runtime);

  detail::TaskContext context{task, variant_kind, *regions};
  auto idx            = context.scalars()[0].value<int32_t>();
  auto value_and_size = ReturnValues::extract(task->futures[0], idx);

  // Legion postamble
  value_and_size.finalize(legion_context);
}

template <LegateVariantCode variant_id>
void register_extract_scalar_variant(const std::unique_ptr<TaskInfo>& task_info)
{
  // TODO(wonchanl): We could support Legion & Realm calling convensions so we don't pass nullptr
  // here
  task_info->add_variant(
    variant_id, nullptr, Legion::CodeDescriptor{extract_scalar_task<variant_id>}, VariantOptions{});
}

}  // namespace

void register_legate_core_tasks(Library* core_lib)
{
  auto task_info = std::make_unique<TaskInfo>("core::extract_scalar");
  register_extract_scalar_variant<LEGATE_CPU_VARIANT>(task_info);
#if LegateDefined(LEGATE_USE_CUDA)
  register_extract_scalar_variant<LEGATE_GPU_VARIANT>(task_info);
#endif
#if LegateDefined(LEGATE_USE_OPENMP)
  register_extract_scalar_variant<LEGATE_OMP_VARIANT>(task_info);
#endif
  core_lib->register_task(LEGATE_CORE_EXTRACT_SCALAR_TASK_ID, std::move(task_info));

  register_array_tasks(core_lib);
  comm::register_tasks(core_lib);
}

#define BUILTIN_REDOP_ID(OP, TYPE_CODE) \
  (LEGION_REDOP_BASE + (OP) * LEGION_TYPE_TOTAL + (static_cast<int32_t>(TYPE_CODE)))

#define RECORD(OP, TYPE_CODE) \
  PrimitiveType(TYPE_CODE).record_reduction_operator(OP, BUILTIN_REDOP_ID(OP, TYPE_CODE));

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
  RECORD_ALL(ADD_LT)
  RECORD(ADD_LT, Type::Code::COMPLEX128)
  RECORD_ALL(SUB_LT)
  RECORD_ALL(MUL_LT)
  RECORD_ALL(DIV_LT)

  RECORD_INT(MAX_LT)
  RECORD_FLOAT(MAX_LT)

  RECORD_INT(MIN_LT)
  RECORD_FLOAT(MIN_LT)

  RECORD_INT(OR_LT)
  RECORD_INT(AND_LT)
  RECORD_INT(XOR_LT)
}

extern void register_exception_reduction_op(const Library* context);

namespace {

void parse_config()
{
  if (!LegateDefined(LEGATE_USE_CUDA)) {
    const char* need_cuda = getenv("LEGATE_NEED_CUDA");
    if (need_cuda != nullptr) {
      // ignore fprintf return values here, we are about to exit anyways
      static_cast<void>(fprintf(stderr,
                                "Legate was run with GPUs but was not built with GPU support. "
                                "Please install Legate again with the \"--cuda\" flag.\n"));
      exit(1);
    }
  }
  if (!LegateDefined(LEGATE_USE_OPENMP)) {
    const char* need_openmp = getenv("LEGATE_NEED_OPENMP");
    if (need_openmp != nullptr) {
      static_cast<void>(
        // TODO(jfaibussowit): Change --openmp -> --with-openmp in build-system update
        fprintf(stderr,
                "Legate was run with OpenMP enabled, but was not built with OpenMP support. "
                "Please install Legate again with the \"--openmp\" flag.\n"));
      exit(1);
    }
  }
  if (!LegateDefined(LEGATE_USE_NETWORK)) {
    const char* need_network = getenv("LEGATE_NEED_NETWORK");
    if (need_network != nullptr) {
      static_cast<void>(
        fprintf(stderr,
                "Legate was run on multiple nodes but was not built with networking "
                "support. Please install Legate again with \"--network\".\n"));
      exit(1);
    }
  }

  auto parse_variable = [](const char* variable, bool& result) {
    const char* value = getenv(variable);
    try {
      if (value != nullptr && safe_strtoll(value) > 0) {
        result = true;
      }
    } catch (const std::exception& excn) {  // thrown by safe_strtoll()
      static_cast<void>(fprintf(stderr, "failed to parse %s: %s\n", variable, excn.what()));
      std::terminate();
    }
  };

  parse_variable("LEGATE_SHOW_PROGRESS", Config::show_progress_requested);
  parse_variable("LEGATE_EMPTY_TASK", Config::use_empty_task);
  parse_variable("LEGATE_SYNC_STREAM_VIEW", Config::synchronize_stream_view);
  parse_variable("LEGATE_LOG_MAPPING", Config::log_mapping_decisions);
  parse_variable("LEGATE_LOG_PARTITIONING", Config::log_partitioning_decisions);
  parse_variable("LEGATE_WARMUP_NCCL", Config::warmup_nccl);
}

}  // namespace

void initialize_core_library()
{
  parse_config();

  ResourceConfig config;
  config.max_tasks       = LEGATE_CORE_MAX_TASK_ID;
  config.max_dyn_tasks   = config.max_tasks - LEGATE_CORE_FIRST_DYNAMIC_TASK_ID;
  config.max_projections = LEGATE_CORE_MAX_FUNCTOR_ID;
  // We register one sharding functor for each new projection functor
  config.max_shardings     = LEGATE_CORE_MAX_FUNCTOR_ID;
  config.max_reduction_ops = LEGATE_CORE_MAX_REDUCTION_OP_ID;

  auto core_lib = Runtime::get_runtime()->create_library(
    CORE_LIBRARY_NAME, config, mapping::detail::create_core_mapper(), true /*in_callback*/);

  register_legate_core_tasks(core_lib);

  register_builtin_reduction_ops();

  register_exception_reduction_op(core_lib);

  register_legate_core_sharding_functors(core_lib);
}

void initialize_core_library_callback(
  Legion::Machine,  // NOLINT(performance-unnecessary-value-param)
  Legion::Runtime*,
  const std::set<Processor>&)
{
  initialize_core_library();
}

// Simple wrapper for variables with default values
template <auto DEFAULT, auto SCALE = decltype(DEFAULT){1}, typename VAL = decltype(DEFAULT)>
class VarWithDefault {
 public:
  [[nodiscard]] VAL value() const { return (has_value() ? value_ : DEFAULT) * SCALE; }
  [[nodiscard]] bool has_value() const { return value_ != UNSET; }
  [[nodiscard]] VAL& ref() { return value_; }

 private:
  static constexpr VAL UNSET{std::numeric_limits<VAL>::max()};
  VAL value_{UNSET};
};

template <typename Runtime, typename Value, Value DEFAULT, Value SCALE>
void try_set_property(Runtime& runtime,
                      const std::string& module_name,
                      const std::string& property_name,
                      const VarWithDefault<DEFAULT, SCALE, Value>& var,
                      const std::string& error_msg)
{
  auto value = var.value();
  if (value < 0) {
    LEGATE_ABORT(error_msg);
  }
  auto config = runtime.get_module_config(module_name);
  if (nullptr == config) {
    // If the variable doesn't have a value, we don't care if the module is nonexistent
    if (!var.has_value()) {
      return;
    }
    LEGATE_ABORT(error_msg << " (the " << module_name << " module is not available)");
  }
  auto success = config->set_property(property_name, value);
  if (!success) {
    LEGATE_ABORT(error_msg);
  }
}

namespace {

constexpr int DEFAULT_CPUS                    = 1;
constexpr int DEFAULT_GPUS                    = 0;
constexpr int DEFAULT_OMPS                    = 0;
constexpr int64_t DEFAULT_OMPTHREADS          = 2;
constexpr int DEFAULT_UTILITY                 = 1;
constexpr int64_t DEFAULT_SYSMEM              = 4000;  // MB
constexpr int64_t DEFAULT_NUMAMEM             = 0;     // MB
constexpr int64_t DEFAULT_FBMEM               = 4000;  // MB
constexpr int64_t DEFAULT_ZCMEM               = 32;    // MB
constexpr int64_t DEFAULT_REGMEM              = 0;     // MB
constexpr int64_t DEFAULT_EAGER_ALLOC_PERCENT = 50;
constexpr int64_t MB                          = 1024 * 1024;

}  // namespace

void handle_legate_args(int32_t argc, char** argv)
{
  // Realm uses ints rather than unsigned ints
  VarWithDefault<DEFAULT_CPUS> cpus;
  VarWithDefault<DEFAULT_GPUS> gpus;
  VarWithDefault<DEFAULT_OMPS> omps;
  VarWithDefault<DEFAULT_OMPTHREADS> ompthreads;
  VarWithDefault<DEFAULT_UTILITY> util;
  VarWithDefault<DEFAULT_SYSMEM, MB> sysmem;
  VarWithDefault<DEFAULT_NUMAMEM, MB> numamem;
  VarWithDefault<DEFAULT_FBMEM, MB> fbmem;
  VarWithDefault<DEFAULT_ZCMEM, MB> zcmem;
  VarWithDefault<DEFAULT_REGMEM, MB> regmem;
  VarWithDefault<DEFAULT_EAGER_ALLOC_PERCENT> eager_alloc_percent;

  Realm::CommandLineParser cp;
  cp.add_option_int("--cpus", cpus.ref())
    .add_option_int("--gpus", gpus.ref())
    .add_option_int("--omps", omps.ref())
    .add_option_int("--ompthreads", ompthreads.ref())
    .add_option_int("--utility", util.ref())
    .add_option_int("--sysmem", sysmem.ref())
    .add_option_int("--numamem", numamem.ref())
    .add_option_int("--fbmem", fbmem.ref())
    .add_option_int("--zcmem", zcmem.ref())
    .add_option_int("--regmem", regmem.ref())
    .add_option_int("--eager-alloc-percentage", eager_alloc_percent.ref())
    .parse_command_line(argc, argv);

  auto rt = Realm::Runtime::get_runtime();

  // ensure core module
  if (!rt.get_module_config("core")) {
    LEGATE_ABORT("core module config is missing");
  }

  // ensure sensible utility
  if (const auto nutil = util.value(); nutil < 1) {
    LEGATE_ABORT("--utility must be at least 1 (have " << nutil << ")");
  }

  // Set core configuration properties
  try_set_property(rt, "core", "cpu", cpus, "unable to set --cpus");
  try_set_property(rt, "core", "util", util, "unable to set --utility");
  try_set_property(rt, "core", "sysmem", sysmem, "unable to set --sysmem");
  try_set_property(rt, "core", "regmem", regmem, "unable to set --regmem");

  // Set CUDA configuration properties
  try_set_property(rt, "cuda", "gpu", gpus, "unable to set --gpus");
  try_set_property(rt, "cuda", "fbmem", fbmem, "unable to set --fbmem");
  try_set_property(rt, "cuda", "zcmem", zcmem, "unable to set --zcmem");

  if (gpus.value() > 0) {
    setenv("LEGATE_NEED_CUDA", "1", true);
  }

  // Set OpenMP configuration properties
  if (omps.value() > 0 && ompthreads.value() == 0) {
    LEGATE_ABORT("--omps configured with zero threads");
  }
  if (omps.value() > 0) {
    setenv("LEGATE_NEED_OPENMP", "1", true);
  }
  try_set_property(rt, "openmp", "ocpu", omps, "unable to set --omps");
  try_set_property(rt, "openmp", "othr", ompthreads, "unable to set --ompthreads");

  // Set NUMA configuration properties
  try_set_property(rt, "numa", "numamem", numamem, "unable to set --numamem");

  std::stringstream ss;

  // eager alloc has to be passed via env var
  ss << "-lg:eager_alloc_percentage " << eager_alloc_percent.value() << " -lg:local 0 ";
  if (const char* existing_default_args = getenv("LEGION_DEFAULT_ARGS")) {
    ss << existing_default_args;
  }
  setenv("LEGION_DEFAULT_ARGS", ss.str().c_str(), true);
}

}  // namespace legate::detail
