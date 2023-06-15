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

#pragma once

#include <memory>
#include <optional>

#include "core/data/logical_store.h"
#include "core/data/shape.h"
#include "core/data/store.h"
#include "core/mapping/machine.h"
#include "core/runtime/detail/communicator_manager.h"
#include "core/runtime/detail/field_manager.h"
#include "core/runtime/detail/machine_manager.h"
#include "core/runtime/detail/partition_manager.h"
#include "core/runtime/detail/provenance_manager.h"
#include "core/runtime/detail/region_manager.h"
#include "core/runtime/resource.h"
#include "core/task/exception.h"
#include "core/type/type_info.h"
#include "core/utilities/multi_set.h"

namespace legate {
class AutoTask;
class Copy;
class LibraryContext;
class ManualTask;
class Operation;
}  // namespace legate

namespace legate::detail {

class LogicalRegionField;
class LogicalStore;

class Runtime {
 public:
  Runtime();
  ~Runtime();

 public:
  LibraryContext* find_library(const std::string& library_name, bool can_fail = false) const;
  LibraryContext* create_library(const std::string& library_name,
                                 const ResourceConfig& config            = ResourceConfig{},
                                 std::unique_ptr<mapping::Mapper> mapper = nullptr);

 public:
  uint32_t get_type_uid();
  void record_reduction_operator(int32_t type_uid, int32_t op_kind, int32_t legion_op_id);
  int32_t find_reduction_operator(int32_t type_uid, int32_t op_kind) const;

 public:
  void initialize(Legion::Context legion_context);

 public:
  mapping::MachineDesc slice_machine_for_task(LibraryContext* library, int64_t task_id);
  std::unique_ptr<AutoTask> create_task(LibraryContext* library, int64_t task_id);
  std::unique_ptr<ManualTask> create_task(LibraryContext* library,
                                          int64_t task_id,
                                          const Shape& launch_shape);
  std::unique_ptr<Copy> create_copy(LibraryContext* library);
  void issue_fill(LibraryContext* library, legate::LogicalStore lhs, legate::LogicalStore value);
  void flush_scheduling_window();
  void submit(std::unique_ptr<Operation> op);

 public:
  std::shared_ptr<LogicalStore> create_store(std::unique_ptr<Type> type, int32_t dim = 1);
  std::shared_ptr<LogicalStore> create_store(const Shape& extents,
                                             std::unique_ptr<Type> type,
                                             bool optimize_scalar = false);
  std::shared_ptr<LogicalStore> create_store(const Scalar& scalar);

 public:
  uint32_t max_pending_exceptions() const;
  void set_max_pending_exceptions(uint32_t max_pending_exceptions);
  void raise_pending_task_exception();
  std::optional<TaskException> check_pending_task_exception();
  void record_pending_exception(const Legion::Future& pending_exception);

 public:
  uint64_t get_unique_store_id();
  uint64_t get_unique_storage_id();

 public:
  std::shared_ptr<LogicalRegionField> create_region_field(const Shape& extents,
                                                          uint32_t field_size);
  std::shared_ptr<LogicalRegionField> import_region_field(Legion::LogicalRegion region,
                                                          Legion::FieldID field_id,
                                                          uint32_t field_size);
  RegionField map_region_field(LibraryContext* context, const LogicalRegionField* region_field);
  void unmap_physical_region(Legion::PhysicalRegion pr);
  size_t num_inline_mapped() const;

 public:
  RegionManager* find_or_create_region_manager(const Legion::Domain& shape);
  FieldManager* find_or_create_field_manager(const Legion::Domain& shape, uint32_t field_size);
  CommunicatorManager* communicator_manager() const;
  MachineManager* machine_manager() const;
  PartitionManager* partition_manager() const;
  ProvenanceManager* provenance_manager() const;

 public:
  Legion::IndexSpace find_or_create_index_space(const Legion::Domain& shape);
  Legion::IndexPartition create_restricted_partition(const Legion::IndexSpace& index_space,
                                                     const Legion::IndexSpace& color_space,
                                                     Legion::PartitionKind kind,
                                                     const Legion::DomainTransform& transform,
                                                     const Legion::Domain& extent);
  Legion::IndexPartition create_weighted_partition(const Legion::IndexSpace& index_space,
                                                   const Legion::IndexSpace& color_space,
                                                   const Legion::FutureMap& weights);
  Legion::FieldSpace create_field_space();
  Legion::LogicalRegion create_region(const Legion::IndexSpace& index_space,
                                      const Legion::FieldSpace& field_space);
  void destroy_region(const Legion::LogicalRegion& logical_region, bool unordered = false);
  Legion::LogicalPartition create_logical_partition(const Legion::LogicalRegion& logical_region,
                                                    const Legion::IndexPartition& index_partition);
  Legion::LogicalRegion get_subregion(const Legion::LogicalPartition& partition,
                                      const Legion::DomainPoint& color);
  Legion::LogicalRegion find_parent_region(const Legion::LogicalRegion& region);
  Legion::Future create_future(const void* data, size_t datalen) const;
  Legion::FieldID allocate_field(const Legion::FieldSpace& field_space, size_t field_size);
  Legion::FieldID allocate_field(const Legion::FieldSpace& field_space,
                                 Legion::FieldID field_id,
                                 size_t field_size);
  Legion::Domain get_index_space_domain(const Legion::IndexSpace& index_space) const;
  Legion::FutureMap delinearize_future_map(const Legion::FutureMap& future_map,
                                           const Legion::IndexSpace& new_domain) const;
  std::pair<Legion::PhaseBarrier, Legion::PhaseBarrier> create_barriers(size_t num_tasks);
  void destroy_barrier(Legion::PhaseBarrier barrier);
  Legion::Future get_tunable(Legion::MapperID mapper_id, int64_t tunable_id, size_t size);
  template <class T>
  T get_tunable(Legion::MapperID mapper_id, int64_t tunable_id)
  {
    return get_tunable(mapper_id, tunable_id, sizeof(T)).get_result<T>();
  }

 public:
  Legion::Future dispatch(Legion::TaskLauncher* launcher,
                          std::vector<Legion::OutputRequirement>* output_requirements = nullptr);
  Legion::FutureMap dispatch(Legion::IndexTaskLauncher* launcher,
                             std::vector<Legion::OutputRequirement>* output_requirements = nullptr);
  void dispatch(Legion::CopyLauncher* launcher);
  void dispatch(Legion::IndexCopyLauncher* launcher);
  void dispatch(Legion::FillLauncher* launcher);
  void dispatch(Legion::IndexFillLauncher* launcher);

 public:
  Legion::Future extract_scalar(const Legion::Future& result, uint32_t idx) const;
  Legion::FutureMap extract_scalar(const Legion::FutureMap& result,
                                   uint32_t idx,
                                   const Legion::Domain& launch_domain) const;
  Legion::Future reduce_future_map(const Legion::FutureMap& future_map, int32_t reduction_op) const;
  Legion::Future reduce_exception_future_map(const Legion::FutureMap& future_map) const;

 public:
  void issue_execution_fence(bool block = false);

 public:
  void initialize_toplevel_machine();
  const mapping::MachineDesc& get_machine() const;

 public:
  Legion::ProjectionID get_projection(int32_t src_ndim, const proj::SymbolicPoint& point);
  Legion::ProjectionID get_delinearizing_projection();
  Legion::ShardingID get_sharding(const mapping::MachineDesc& machine,
                                  Legion::ProjectionID proj_id);

 private:
  void schedule(std::vector<std::unique_ptr<Operation>> operations);

 public:
  static Runtime* get_runtime();
  static int32_t start(int32_t argc, char** argv);
  bool initialized() const { return initialized_; }
  void destroy();
  int32_t finish();

 private:
  bool initialized_{false};
  Legion::Runtime* legion_runtime_{nullptr};
  Legion::Context legion_context_{nullptr};
  LibraryContext* core_context_{nullptr};

 private:
  using FieldManagerKey = std::pair<Legion::Domain, uint32_t>;
  std::map<FieldManagerKey, FieldManager*> field_managers_;
  std::map<Legion::Domain, RegionManager*> region_managers_;
  CommunicatorManager* communicator_manager_{nullptr};
  MachineManager* machine_manager_{nullptr};
  PartitionManager* partition_manager_{nullptr};
  ProvenanceManager* provenance_manager_{nullptr};

 private:
  std::map<Legion::Domain, Legion::IndexSpace> index_spaces_;

 private:
  using ProjectionDesc = std::pair<int32_t, proj::SymbolicPoint>;
  int64_t next_projection_id_{LEGATE_CORE_FIRST_DYNAMIC_FUNCTOR_ID};
  std::map<ProjectionDesc, Legion::ProjectionID> registered_projections_{};

 private:
  using ShardingDesc = std::tuple<Legion::ProjectionID, uint32_t, uint32_t, uint32_t, uint32_t>;
  int64_t next_sharding_id_{LEGATE_CORE_FIRST_DYNAMIC_FUNCTOR_ID};
  std::map<ShardingDesc, Legion::ShardingID> registered_shardings_{};

 private:
  std::vector<std::unique_ptr<Operation>> operations_;
  size_t window_size_{1};
  uint64_t next_unique_id_{0};

 private:
  using RegionFieldID = std::pair<Legion::LogicalRegion, Legion::FieldID>;
  std::map<RegionFieldID, Legion::PhysicalRegion> inline_mapped_;
  MultiSet<Legion::PhysicalRegion> physical_region_refs_;
  uint64_t next_store_id_{1};
  uint64_t next_storage_id_{1};

 private:
  std::map<std::string, LibraryContext*> libraries_{};

 private:
  uint32_t next_type_uid_;
  std::map<std::pair<int32_t, int32_t>, int32_t> reduction_ops_{};

 private:
  uint32_t max_pending_exceptions_;
  std::vector<Legion::Future> pending_exceptions_{};
  std::deque<TaskException> outstanding_exceptions_{};
};

void registration_callback(Legion::Machine machine,
                           Legion::Runtime* legion_runtime,
                           const std::set<Processor>& local_procs);

void registration_callback_for_python(Legion::Machine machine,
                                      Legion::Runtime* legion_runtime,
                                      const std::set<Processor>& local_procs);

}  // namespace legate::detail
