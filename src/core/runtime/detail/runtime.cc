/* Copyright 2023 NVIDIA Corporation
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

#include "core/runtime/detail/runtime.h"

#include "core/comm/comm.h"
#include "core/data/detail/logical_region_field.h"
#include "core/data/detail/logical_store.h"
#include "core/mapping/core_mapper.h"
#include "core/mapping/default_mapper.h"
#include "core/partitioning/partitioner.h"
#include "core/runtime/detail/fill.h"
#include "core/runtime/detail/task_launcher.h"
#include "core/runtime/projection.h"
#include "core/runtime/shard.h"
#include "env_defaults.h"

#include "realm/network.h"

namespace legate {

Logger log_legate("legate");

const char* const core_library_name = "legate.core";

}  // namespace legate

namespace legate::detail {

namespace {

constexpr uint32_t CUSTOM_TYPE_UID_BASE = 1000;
const char* TOPLEVEL_NAME               = "Legate Core Toplevel Task";

}  // namespace

Runtime::Runtime()
  : legion_runtime_(Legion::Runtime::get_runtime()),
    next_type_uid_(CUSTOM_TYPE_UID_BASE),
    max_pending_exceptions_(extract_env(
      "LEGATE_MAX_PENDING_EXCEPTIONS", MAX_PENDING_EXCEPTIONS_DEFAULT, MAX_PENDING_EXCEPTIONS_TEST))
{
}

Runtime::~Runtime() {}

LibraryContext* Runtime::find_library(const std::string& library_name,
                                      bool can_fail /*=false*/) const
{
  auto finder = libraries_.find(library_name);
  if (libraries_.end() == finder) {
    if (!can_fail) throw std::out_of_range("Library " + library_name + " does not exist");
    return nullptr;
  }
  return finder->second;
}

LibraryContext* Runtime::create_library(const std::string& library_name,
                                        const ResourceConfig& config,
                                        std::unique_ptr<mapping::Mapper> mapper)
{
  if (libraries_.find(library_name) != libraries_.end())
    throw std::invalid_argument("Library " + library_name + " already exists");

  log_legate.debug("Library %s is created", library_name.c_str());
  if (nullptr == mapper) mapper = std::make_unique<mapping::DefaultMapper>();
  auto context             = new LibraryContext(library_name, config, std::move(mapper));
  libraries_[library_name] = context;
  return context;
}

uint32_t Runtime::get_type_uid() { return next_type_uid_++; }

void Runtime::record_reduction_operator(int32_t type_uid, int32_t op_kind, int32_t legion_op_id)
{
#ifdef DEBUG_LEGATE
  log_legate.debug("Record reduction op (type_uid: %d, op_kind: %d, legion_op_id: %d)",
                   type_uid,
                   op_kind,
                   legion_op_id);
#endif
  auto key    = std::make_pair(type_uid, op_kind);
  auto finder = reduction_ops_.find(key);
  if (finder != reduction_ops_.end()) {
    std::stringstream ss;
    ss << "Reduction op " << op_kind << " already exists for type " << type_uid;
    throw std::invalid_argument(std::move(ss).str());
  }
  reduction_ops_.emplace(std::make_pair(key, legion_op_id));
}

int32_t Runtime::find_reduction_operator(int32_t type_uid, int32_t op_kind) const
{
  auto key    = std::make_pair(type_uid, op_kind);
  auto finder = reduction_ops_.find(key);
  if (reduction_ops_.end() == finder) {
#ifdef DEBUG_LEGATE
    log_legate.debug("Can't find reduction op (type_uid: %d, op_kind: %d)", type_uid, op_kind);
#endif
    std::stringstream ss;
    ss << "Reduction op " << op_kind << " does not exist for type " << type_uid;
    throw std::invalid_argument(std::move(ss).str());
  }
#ifdef DEBUG_LEGATE
  log_legate.debug(
    "Found reduction op %d (type_uid: %d, op_kind: %d)", finder->second, type_uid, op_kind);
#endif
  return finder->second;
}

void Runtime::initialize(Legion::Context legion_context)
{
  if (initialized_) throw std::runtime_error("Legate runtime has already been initialized");
  initialized_          = true;
  legion_context_       = legion_context;
  core_context_         = find_library(core_library_name);
  communicator_manager_ = new CommunicatorManager();
  partition_manager_    = new PartitionManager(this, core_context_);
  machine_manager_      = new MachineManager();
  provenance_manager_   = new ProvenanceManager();
  Core::retrieve_tunable(legion_context_, legion_runtime_, core_context_);
  initialize_toplevel_machine();
  comm::register_builtin_communicator_factories(core_context_);
}

mapping::MachineDesc Runtime::slice_machine_for_task(LibraryContext* library, int64_t task_id)
{
  auto task_info = library->find_task(task_id);
  if (nullptr == task_info) {
    std::stringstream ss;
    ss << "Library " << library->get_library_name() << " does not have task " << task_id;
    throw std::invalid_argument(std::move(ss).str());
  }

  std::vector<mapping::TaskTarget> task_targets;
  auto& machine = machine_manager_->get_machine();
  for (const auto& t : machine.valid_targets()) {
    if (task_info->has_variant(mapping::to_variant_code(t))) task_targets.push_back(t);
  }
  auto sliced = machine.only(task_targets);

  if (sliced.empty()) {
    std::stringstream ss;
    ss << "Task " << task_id << " (" << task_info->name() << ") of library "
       << library->get_library_name() << " does not have any valid variant for "
       << "the current machine configuration.";
    throw std::invalid_argument(ss.str());
  }
  return sliced;
}

// This function should be moved to the library context
std::unique_ptr<AutoTask> Runtime::create_task(LibraryContext* library, int64_t task_id)
{
  auto machine = slice_machine_for_task(library, task_id);
  auto task    = new AutoTask(library, task_id, next_unique_id_++, std::move(machine));
  return std::unique_ptr<AutoTask>(task);
}

std::unique_ptr<ManualTask> Runtime::create_task(LibraryContext* library,
                                                 int64_t task_id,
                                                 const Shape& launch_shape)
{
  auto machine = slice_machine_for_task(library, task_id);
  auto task = new ManualTask(library, task_id, launch_shape, next_unique_id_++, std::move(machine));
  return std::unique_ptr<ManualTask>(task);
}

std::unique_ptr<Copy> Runtime::create_copy(LibraryContext* library)
{
  auto machine = machine_manager_->get_machine();
  auto copy    = new Copy(library, next_unique_id_++, std::move(machine));
  return std::unique_ptr<Copy>(copy);
}

void Runtime::issue_fill(LibraryContext* library,
                         legate::LogicalStore lhs,
                         legate::LogicalStore value)
{
  auto machine = machine_manager_->get_machine();
  submit(
    std::unique_ptr<Fill>(new Fill(library, lhs, value, next_unique_id_++, std::move(machine))));
}

void Runtime::flush_scheduling_window()
{
  if (operations_.size() == 0) return;

  std::vector<std::unique_ptr<Operation>> to_schedule;
  to_schedule.swap(operations_);
  schedule(std::move(to_schedule));
}

void Runtime::submit(std::unique_ptr<Operation> op)
{
  operations_.push_back(std::move(op));
  if (operations_.size() >= window_size_) { flush_scheduling_window(); }
}

void Runtime::schedule(std::vector<std::unique_ptr<Operation>> operations)
{
  std::vector<Operation*> op_pointers{};
  op_pointers.reserve(operations.size());
  for (auto& op : operations) op_pointers.push_back(op.get());

  Partitioner partitioner(std::move(op_pointers));
  auto strategy = partitioner.partition_stores();

  for (auto& op : operations) op->launch(strategy.get());
}

std::shared_ptr<LogicalStore> Runtime::create_store(std::unique_ptr<Type> type, int32_t dim)
{
  auto storage = std::make_shared<detail::Storage>(dim, std::move(type));
  return std::make_shared<LogicalStore>(std::move(storage));
}

std::shared_ptr<LogicalStore> Runtime::create_store(const Shape& extents,
                                                    std::unique_ptr<Type> type,
                                                    bool optimize_scalar /*=false*/)
{
  auto storage = std::make_shared<detail::Storage>(extents, std::move(type), optimize_scalar);
  return std::make_shared<detail::LogicalStore>(std::move(storage));
}

std::shared_ptr<LogicalStore> Runtime::create_store(const Scalar& scalar)
{
  Shape extents{1};
  auto future  = create_future(scalar.ptr(), scalar.size());
  auto storage = std::make_shared<detail::Storage>(extents, scalar.type().clone(), future);
  return std::make_shared<detail::LogicalStore>(std::move(storage));
}

uint32_t Runtime::max_pending_exceptions() const { return max_pending_exceptions_; }

void Runtime::set_max_pending_exceptions(uint32_t max_pending_exceptions)
{
  uint32_t old_value      = max_pending_exceptions_;
  max_pending_exceptions_ = max_pending_exceptions;
  if (old_value > max_pending_exceptions_) raise_pending_task_exception();
}

void Runtime::raise_pending_task_exception()
{
  auto exn = check_pending_task_exception();
  if (exn.has_value()) throw exn.value();
}

std::optional<TaskException> Runtime::check_pending_task_exception()
{
  // If there's already an outstanding exception from the previous scan, we just return that.
  if (!outstanding_exceptions_.empty()) {
    std::optional<TaskException> result = outstanding_exceptions_.front();
    outstanding_exceptions_.pop_front();
    return result;
  }

  // Othrewise, we unpack all pending exceptions and push them to the outstanding exception queue
  for (auto& pending_exception : pending_exceptions_) {
    auto returned_exception = pending_exception.get_result<ReturnedException>();
    auto result             = returned_exception.to_task_exception();
    if (result.has_value()) outstanding_exceptions_.push_back(result.value());
  }
  pending_exceptions_.clear();
  return outstanding_exceptions_.empty() ? std::nullopt : check_pending_task_exception();
}

void Runtime::record_pending_exception(const Legion::Future& pending_exception)
{
  pending_exceptions_.push_back(pending_exception);
  if (outstanding_exceptions_.size() + pending_exceptions_.size() >= max_pending_exceptions_)
    raise_pending_task_exception();
}

uint64_t Runtime::get_unique_store_id() { return next_store_id_++; }

uint64_t Runtime::get_unique_storage_id() { return next_storage_id_++; }

std::shared_ptr<LogicalRegionField> Runtime::create_region_field(const Shape& extents,
                                                                 uint32_t field_size)
{
  DomainPoint lo, hi;
  hi.dim = lo.dim = static_cast<int32_t>(extents.size());
  assert(lo.dim <= LEGION_MAX_DIM);
  for (int32_t dim = 0; dim < lo.dim; ++dim) lo[dim] = 0;
  for (int32_t dim = 0; dim < lo.dim; ++dim) hi[dim] = extents[dim] - 1;

  Domain shape(lo, hi);
  auto fld_mgr = find_or_create_field_manager(shape, field_size);
  return fld_mgr->allocate_field();
}

std::shared_ptr<LogicalRegionField> Runtime::import_region_field(Legion::LogicalRegion region,
                                                                 Legion::FieldID field_id,
                                                                 uint32_t field_size)
{
  // TODO: This is a blocking operation. We should instead use index spaces as keys to field
  // managers
  auto shape   = legion_runtime_->get_index_space_domain(legion_context_, region.get_index_space());
  auto fld_mgr = find_or_create_field_manager(shape, field_size);
  return fld_mgr->import_field(region, field_id);
}

RegionField Runtime::map_region_field(LibraryContext* context, const LogicalRegionField* rf)
{
  auto root_region = rf->get_root().region();
  auto field_id    = rf->field_id();

  Legion::PhysicalRegion pr;

  RegionFieldID key(root_region, field_id);
  auto finder = inline_mapped_.find(key);
  if (inline_mapped_.end() == finder) {
    Legion::RegionRequirement req(root_region, READ_WRITE, EXCLUSIVE, root_region);
    req.add_field(field_id);

    auto mapper_id = context->get_mapper_id();
    // TODO: We need to pass the metadata about logical store
    Legion::InlineLauncher launcher(req, mapper_id);
    pr                  = legion_runtime_->map_region(legion_context_, launcher);
    inline_mapped_[key] = pr;
  } else
    pr = finder->second;
  physical_region_refs_.add(pr);
  return RegionField(rf->dim(), pr, field_id);
}

void Runtime::unmap_physical_region(Legion::PhysicalRegion pr)
{
  if (physical_region_refs_.remove(pr)) {
    // The last user of this inline mapping was removed, so remove it from our cache and unmap.
    std::vector<Legion::FieldID> fields;
    pr.get_fields(fields);
    assert(fields.size() == 1);
    RegionFieldID key(pr.get_logical_region(), fields[0]);
    auto finder = inline_mapped_.find(key);
    assert(finder != inline_mapped_.end() && finder->second == pr);
    inline_mapped_.erase(finder);
    legion_runtime_->unmap_region(legion_context_, pr);
  }
}

size_t Runtime::num_inline_mapped() const { return inline_mapped_.size(); }

RegionManager* Runtime::find_or_create_region_manager(const Domain& shape)
{
  auto finder = region_managers_.find(shape);
  if (finder != region_managers_.end())
    return finder->second;
  else {
    auto rgn_mgr            = new RegionManager(this, shape);
    region_managers_[shape] = rgn_mgr;
    return rgn_mgr;
  }
}

FieldManager* Runtime::find_or_create_field_manager(const Domain& shape, uint32_t field_size)
{
  auto key    = FieldManagerKey(shape, field_size);
  auto finder = field_managers_.find(key);
  if (finder != field_managers_.end())
    return finder->second;
  else {
    auto fld_mgr         = new FieldManager(this, shape, field_size);
    field_managers_[key] = fld_mgr;
    return fld_mgr;
  }
}

PartitionManager* Runtime::partition_manager() const { return partition_manager_; }

ProvenanceManager* Runtime::provenance_manager() const { return provenance_manager_; }

Legion::IndexSpace Runtime::find_or_create_index_space(const Domain& shape)
{
  assert(nullptr != legion_context_);
  auto finder = index_spaces_.find(shape);
  if (finder != index_spaces_.end())
    return finder->second;
  else {
    auto is              = legion_runtime_->create_index_space(legion_context_, shape);
    index_spaces_[shape] = is;
    return is;
  }
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

Legion::FieldSpace Runtime::create_field_space()
{
  assert(nullptr != legion_context_);
  return legion_runtime_->create_field_space(legion_context_);
}

Legion::LogicalRegion Runtime::create_region(const Legion::IndexSpace& index_space,
                                             const Legion::FieldSpace& field_space)
{
  assert(nullptr != legion_context_);
  return legion_runtime_->create_logical_region(legion_context_, index_space, field_space);
}

void Runtime::destroy_region(const Legion::LogicalRegion& logical_region, bool unordered)
{
  assert(nullptr != legion_context_);
  legion_runtime_->destroy_logical_region(legion_context_, logical_region, unordered);
}

Legion::LogicalPartition Runtime::create_logical_partition(
  const Legion::LogicalRegion& logical_region, const Legion::IndexPartition& index_partition)
{
  assert(nullptr != legion_context_);
  return legion_runtime_->get_logical_partition(legion_context_, logical_region, index_partition);
}

Legion::LogicalRegion Runtime::get_subregion(const Legion::LogicalPartition& partition,
                                             const Legion::DomainPoint& color)
{
  assert(nullptr != legion_context_);
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

Legion::Future Runtime::create_future(const void* data, size_t datalen) const
{
  return Legion::Future::from_untyped_pointer(data, datalen);
}

Legion::FieldID Runtime::allocate_field(const Legion::FieldSpace& field_space, size_t field_size)
{
  assert(nullptr != legion_context_);
  auto allocator = legion_runtime_->create_field_allocator(legion_context_, field_space);
  return allocator.allocate_field(field_size);
}

Legion::FieldID Runtime::allocate_field(const Legion::FieldSpace& field_space,
                                        Legion::FieldID field_id,
                                        size_t field_size)
{
  assert(nullptr != legion_context_);
  auto allocator = legion_runtime_->create_field_allocator(legion_context_, field_space);
  return allocator.allocate_field(field_size, field_id);
}

Domain Runtime::get_index_space_domain(const Legion::IndexSpace& index_space) const
{
  assert(nullptr != legion_context_);
  return legion_runtime_->get_index_space_domain(legion_context_, index_space);
}

namespace {

Legion::DomainPoint _delinearize_future_map(const DomainPoint& point,
                                            const Domain& domain,
                                            const Domain& range)
{
  assert(range.dim == 1);
  DomainPoint result;
  result.dim = 1;

  int32_t ndim = domain.dim;
  int64_t idx  = point[0];
  for (int32_t dim = 1; dim < ndim; ++dim) {
    int64_t extent = domain.rect_data[dim + ndim] - domain.rect_data[dim] + 1;
    idx            = idx * extent + point[dim];
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
  return std::make_pair(arrival_barrier, wait_barrier);
}

void Runtime::destroy_barrier(Legion::PhaseBarrier barrier)
{
  legion_runtime_->destroy_phase_barrier(legion_context_, barrier);
}

Legion::Future Runtime::get_tunable(Legion::MapperID mapper_id, int64_t tunable_id, size_t size)
{
  Legion::TunableLauncher launcher(tunable_id, mapper_id, 0, size);
  return legion_runtime_->select_tunable_value(legion_context_, launcher);
}

Legion::Future Runtime::dispatch(Legion::TaskLauncher* launcher,
                                 std::vector<Legion::OutputRequirement>* output_requirements)
{
  assert(nullptr != legion_context_);
  return legion_runtime_->execute_task(legion_context_, *launcher, output_requirements);
}

Legion::FutureMap Runtime::dispatch(Legion::IndexTaskLauncher* launcher,
                                    std::vector<Legion::OutputRequirement>* output_requirements)
{
  assert(nullptr != legion_context_);
  return legion_runtime_->execute_index_space(legion_context_, *launcher, output_requirements);
}

void Runtime::dispatch(Legion::CopyLauncher* launcher)
{
  assert(nullptr != legion_context_);
  return legion_runtime_->issue_copy_operation(legion_context_, *launcher);
}

void Runtime::dispatch(Legion::IndexCopyLauncher* launcher)
{
  assert(nullptr != legion_context_);
  return legion_runtime_->issue_copy_operation(legion_context_, *launcher);
}

void Runtime::dispatch(Legion::FillLauncher* launcher)
{
  assert(nullptr != legion_context_);
  return legion_runtime_->fill_fields(legion_context_, *launcher);
}

void Runtime::dispatch(Legion::IndexFillLauncher* launcher)
{
  assert(nullptr != legion_context_);
  return legion_runtime_->fill_fields(legion_context_, *launcher);
}

Legion::Future Runtime::extract_scalar(const Legion::Future& result, uint32_t idx) const
{
  auto& machine    = get_machine();
  auto& provenance = provenance_manager()->get_provenance();
  auto variant     = mapping::to_variant_code(machine.preferred_target);
  TaskLauncher launcher(
    core_context_, machine, provenance, LEGATE_CORE_EXTRACT_SCALAR_TASK_ID, variant);
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
  auto variant     = mapping::to_variant_code(machine.preferred_target);
  TaskLauncher launcher(
    core_context_, machine, provenance, LEGATE_CORE_EXTRACT_SCALAR_TASK_ID, variant);
  launcher.add_future_map(result);
  launcher.add_scalar(Scalar(idx));
  return launcher.execute(launch_domain);
}

Legion::Future Runtime::reduce_future_map(const Legion::FutureMap& future_map,
                                          int32_t reduction_op) const
{
  return legion_runtime_->reduce_future_map(legion_context_,
                                            future_map,
                                            reduction_op,
                                            false /*deterministic*/,
                                            core_context_->get_mapper_id());
}

Legion::Future Runtime::reduce_exception_future_map(const Legion::FutureMap& future_map) const
{
  auto reduction_op = core_context_->get_reduction_op_id(LEGATE_CORE_JOIN_EXCEPTION_OP);
  return legion_runtime_->reduce_future_map(legion_context_,
                                            future_map,
                                            reduction_op,
                                            false /*deterministic*/,
                                            core_context_->get_mapper_id(),
                                            LEGATE_CORE_JOIN_EXCEPTION_TAG);
}

void Runtime::issue_execution_fence(bool block /*=false*/)
{
  flush_scheduling_window();
  // FIXME: This needs to be a Legate operation
  auto future = legion_runtime_->issue_execution_fence(legion_context_);
  if (block) future.wait();
}

void Runtime::initialize_toplevel_machine()
{
  auto mapper_id = core_context_->get_mapper_id();
  auto num_nodes = get_tunable<int32_t>(mapper_id, LEGATE_CORE_TUNABLE_NUM_NODES);

  auto num_gpus = get_tunable<int32_t>(mapper_id, LEGATE_CORE_TUNABLE_TOTAL_GPUS);
  auto num_omps = get_tunable<int32_t>(mapper_id, LEGATE_CORE_TUNABLE_TOTAL_OMPS);
  auto num_cpus = get_tunable<int32_t>(mapper_id, LEGATE_CORE_TUNABLE_TOTAL_CPUS);

  auto create_range = [&num_nodes](int32_t num_procs) {
    auto per_node_count = num_procs / num_nodes;
    return mapping::ProcessorRange(0, num_procs, per_node_count);
  };

  mapping::MachineDesc machine({{mapping::TaskTarget::GPU, create_range(num_gpus)},
                                {mapping::TaskTarget::OMP, create_range(num_omps)},
                                {mapping::TaskTarget::CPU, create_range(num_cpus)}});
#ifdef DEBUG_LEGATE
  assert(machine_manager_ != nullptr);
#endif

  machine_manager_->push_machine(std::move(machine));
}

const mapping::MachineDesc& Runtime::get_machine() const
{
#ifdef DEBUG_LEGATE
  assert(machine_manager_ != nullptr);
#endif
  return machine_manager_->get_machine();
}

Legion::ProjectionID Runtime::get_projection(int32_t src_ndim, const proj::SymbolicPoint& point)
{
#ifdef DEBUG_LEGATE
  log_legate.debug() << "Query projection {src_ndim: " << src_ndim << ", point: " << point << "}";
#endif

  if (is_identity(src_ndim, point)) {
#ifdef DEBUG_LEGATE
    log_legate.debug() << "Identity projection {src_ndim: " << src_ndim << ", point: " << point
                       << "}";
#endif
    return 0;
  }

  ProjectionDesc key(src_ndim, point);
  auto finder = registered_projections_.find(key);
  if (registered_projections_.end() != finder) return finder->second;

  auto proj_id = core_context_->get_projection_id(next_projection_id_++);

  auto ndim = point.size();
  std::vector<int32_t> dims;
  std::vector<int32_t> weights;
  std::vector<int32_t> offsets;
  for (auto& expr : point.data()) {
    dims.push_back(expr.dim());
    weights.push_back(expr.weight());
    offsets.push_back(expr.offset());
  }
  legate_register_affine_projection_functor(
    src_ndim, ndim, dims.data(), weights.data(), offsets.data(), proj_id);
  registered_projections_[key] = proj_id;

#ifdef DEBUG_LEGATE
  log_legate.debug() << "Register projection " << proj_id << " {src_ndim: " << src_ndim
                     << ", point: " << point << "}";
#endif

  return proj_id;
}

Legion::ProjectionID Runtime::get_delinearizing_projection()
{
  return core_context_->get_projection_id(LEGATE_CORE_DELINEARIZE_PROJ_ID);
}

Legion::ShardingID Runtime::get_sharding(const mapping::MachineDesc& machine,
                                         Legion::ProjectionID proj_id)
{
  // If we're running on a single node, we don't need to generate sharding functors
  if (Realm::Network::max_node_id == 0) return 0;

  auto& proc_range = machine.processor_range();
  auto [low, high] = proc_range.get_node_range();
  auto offset      = proc_range.low % proc_range.per_node_count;
  ShardingDesc key{proj_id, low, high, offset, proc_range.per_node_count};

#ifdef DEBUG_LEGATE
  log_legate.debug() << "Query sharding {proj_id: " << proj_id
                     << ", processor range: " << proc_range
                     << ", processor type: " << machine.preferred_target << "}";
#endif

  auto finder = registered_shardings_.find(key);
  if (finder != registered_shardings_.end()) {
#ifdef DEBUG_LEGATE
    log_legate.debug() << "Found sharding " << finder->second;
#endif
    return finder->second;
  }

  auto sharding_id = core_context_->get_sharding_id(next_sharding_id_++);
  registered_shardings_.insert({key, sharding_id});

#ifdef DEBUG_LEGATE
  log_legate.debug() << "Create sharding " << sharding_id;
#endif

  legate_create_sharding_functor_using_projection(
    sharding_id, proj_id, low, high, offset, proc_range.per_node_count);

  return sharding_id;
}

CommunicatorManager* Runtime::communicator_manager() const { return communicator_manager_; }

MachineManager* Runtime::machine_manager() const { return machine_manager_; }

/*static*/ int32_t Runtime::start(int32_t argc, char** argv)
{
  Legion::Runtime::initialize(&argc, &argv, true /*filter legion and realm args*/);

  Legion::Runtime::add_registration_callback(registration_callback);

  auto result = Legion::Runtime::start(argc, argv, true);
  if (result != 0) {
    log_legate.error("Legion Runtime failed to start.");
    return result;
  }

  // Get the runtime now that we've started it
  auto legion_runtime = Legion::Runtime::get_runtime();

  // Then we can make this thread into an implicit top-level task
  auto legion_context = legion_runtime->begin_implicit_task(LEGATE_CORE_TOPLEVEL_TASK_ID,
                                                            0 /*mapper id*/,
                                                            Processor::LOC_PROC,
                                                            TOPLEVEL_NAME,
                                                            true /*control replicable*/);

  // We can now initialize the Legate runtime with the Legion context
  Runtime::get_runtime()->initialize(legion_context);

  return result;
}

int32_t Runtime::finish()
{
  destroy();

  // Mark that we are done excecuting the top-level task
  // After this call the context is no longer valid
  Legion::Runtime::get_runtime()->finish_implicit_task(legion_context_);

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
  // Flush any outstanding operations before we tear down the runtime
  flush_scheduling_window();

  // Need a fence to make sure all client operations come before the subsequent clean-up tasks
  issue_execution_fence();

  // Destroy all communicators. This will likely launch some tasks for the clean-up.
  communicator_manager_->destroy();

  // Destroy all Legion handles used by Legate
  for (auto& [_, region_manager] : region_managers_) region_manager->destroy(true /*unordered*/);
  for (auto& [_, index_space] : index_spaces_)
    legion_runtime_->destroy_index_space(legion_context_, index_space, true /*unordered*/);
  index_spaces_.clear();

  // We're about to deallocate objects below, so let's block on all outstanding Legion operations
  issue_execution_fence(true);

  // We finally deallocate managers
  for (auto& [_, context] : libraries_) delete context;
  libraries_.clear();
  for (auto& [_, region_manager] : region_managers_) delete region_manager;
  region_managers_.clear();
  for (auto& [_, field_manager] : field_managers_) delete field_manager;
  field_managers_.clear();

  delete communicator_manager_;
  delete machine_manager_;
  delete partition_manager_;
  delete provenance_manager_;
}

static void extract_scalar_task(
  const void* args, size_t arglen, const void* userdata, size_t userlen, Legion::Processor p)
{
  // Legion preamble
  const Legion::Task* task;
  const std::vector<Legion::PhysicalRegion>* regions;
  Legion::Context legion_context;
  Legion::Runtime* runtime;
  Legion::Runtime::legion_task_preamble(args, arglen, p, task, regions, legion_context, runtime);

  Core::show_progress(task, legion_context, runtime);

  TaskContext context(task, *regions, legion_context, runtime);
  auto idx            = context.scalars()[0].value<int32_t>();
  auto value_and_size = ReturnValues::extract(task->futures[0], idx);

  // Legion postamble
  value_and_size.finalize(legion_context);
}

void register_legate_core_tasks(Legion::Machine machine,
                                Legion::Runtime* runtime,
                                const LibraryContext* context)
{
  auto extract_scalar_task_id          = context->get_task_id(LEGATE_CORE_EXTRACT_SCALAR_TASK_ID);
  const char* extract_scalar_task_name = "core::extract_scalar";
  runtime->attach_name(
    extract_scalar_task_id, extract_scalar_task_name, false /*mutable*/, true /*local only*/);

  auto make_registrar = [&](auto task_id, auto* task_name, auto proc_kind) {
    Legion::TaskVariantRegistrar registrar(task_id, task_name);
    registrar.add_constraint(Legion::ProcessorConstraint(proc_kind));
    registrar.set_leaf(true);
    registrar.global_registration = false;
    return registrar;
  };

  // Register the task variants
  auto register_extract_scalar = [&](auto proc_kind, auto variant_id) {
    auto registrar = make_registrar(extract_scalar_task_id, extract_scalar_task_name, proc_kind);
    Legion::CodeDescriptor desc(extract_scalar_task);
    runtime->register_task_variant(
      registrar, desc, nullptr, 0, LEGATE_MAX_SIZE_SCALAR_RETURN, variant_id);
  };
  register_extract_scalar(Processor::LOC_PROC, LEGATE_CPU_VARIANT);
#ifdef LEGATE_USE_CUDA
  register_extract_scalar(Processor::TOC_PROC, LEGATE_GPU_VARIANT);
#endif
#ifdef LEGATE_USE_OPENMP
  register_extract_scalar(Processor::OMP_PROC, LEGATE_OMP_VARIANT);
#endif
  comm::register_tasks(machine, runtime, context);
}

#define BUILTIN_REDOP_ID(OP, TYPE_CODE) \
  (LEGION_REDOP_BASE + (OP)*LEGION_TYPE_TOTAL + (static_cast<int32_t>(TYPE_CODE)))

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

extern void register_exception_reduction_op(Legion::Runtime* runtime,
                                            const LibraryContext* context);

void core_library_registration(Legion::Machine machine,
                               Legion::Runtime* legion_runtime,
                               const std::set<Processor>& local_procs)
{
  ResourceConfig config;
  config.max_tasks       = LEGATE_CORE_NUM_TASK_IDS;
  config.max_projections = LEGATE_CORE_MAX_FUNCTOR_ID;
  // We register one sharding functor for each new projection functor
  config.max_shardings     = LEGATE_CORE_MAX_FUNCTOR_ID;
  config.max_reduction_ops = LEGATE_CORE_MAX_REDUCTION_OP_ID;

  auto runtime  = Runtime::get_runtime();
  auto core_lib = runtime->create_library(core_library_name, config);

  register_legate_core_tasks(machine, legion_runtime, core_lib);

  register_legate_core_mapper(machine, legion_runtime, core_lib);

  register_builtin_reduction_ops();

  register_exception_reduction_op(legion_runtime, core_lib);

  register_legate_core_projection_functors(legion_runtime, core_lib);

  register_legate_core_sharding_functors(legion_runtime, core_lib);
}

void registration_callback(Legion::Machine machine,
                           Legion::Runtime* legion_runtime,
                           const std::set<Processor>& local_procs)
{
  core_library_registration(machine, legion_runtime, local_procs);

  Core::parse_config();
}

void registration_callback_for_python(Legion::Machine machine,
                                      Legion::Runtime* legion_runtime,
                                      const std::set<Processor>& local_procs)
{
  core_library_registration(machine, legion_runtime, local_procs);

  Runtime::get_runtime()->initialize(Legion::Runtime::get_context());
}

}  // namespace legate::detail
