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

#pragma once

#include <memory>
#include <optional>

#include "core/data/detail/scalar.h"
#include "core/data/detail/store.h"
#include "core/data/logical_store.h"
#include "core/data/shape.h"
#include "core/mapping/machine.h"
#include "core/runtime/detail/communicator_manager.h"
#include "core/runtime/detail/consensus_match_result.h"
#include "core/runtime/detail/field_manager.h"
#include "core/runtime/detail/library.h"
#include "core/runtime/detail/machine_manager.h"
#include "core/runtime/detail/partition_manager.h"
#include "core/runtime/detail/projection.h"
#include "core/runtime/detail/provenance_manager.h"
#include "core/runtime/detail/region_manager.h"
#include "core/runtime/resource.h"
#include "core/task/exception.h"
#include "core/type/type_info.h"
#include "core/utilities/multi_set.h"

namespace legate::detail {

class AutoTask;
class BaseLogicalArray;
class Copy;
class Library;
struct LogicalArray;
class LogicalRegionField;
class LogicalStore;
class ManualTask;
class Operation;
class StructLogicalArray;

struct Config {
  static bool show_progress_requested;
  static bool use_empty_task;
  static bool synchronize_stream_view;
  static bool log_mapping_decisions;
  static bool log_partitioning_decisions;
  static bool has_socket_mem;
  static bool warmup_nccl;
};

class Runtime {
 public:
  Runtime();
  ~Runtime();

 public:
  Library* create_library(const std::string& library_name,
                          const ResourceConfig& config,
                          std::unique_ptr<mapping::Mapper> mapper,
                          bool in_callback);
  Library* find_library(const std::string& library_name, bool can_fail) const;
  Library* find_or_create_library(const std::string& library_name,
                                  const ResourceConfig& config,
                                  std::unique_ptr<mapping::Mapper> mapper,
                                  bool* created,
                                  bool in_callback);

 public:
  void record_reduction_operator(int32_t type_uid, int32_t op_kind, int32_t legion_op_id);
  int32_t find_reduction_operator(int32_t type_uid, int32_t op_kind) const;

 public:
  void initialize(Legion::Context legion_context);

 public:
  mapping::detail::Machine slice_machine_for_task(const Library* library, int64_t task_id);
  std::unique_ptr<AutoTask> create_task(const Library* library, int64_t task_id);
  std::unique_ptr<ManualTask> create_task(const Library* library,
                                          int64_t task_id,
                                          const Shape& launch_shape);
  void issue_copy(std::shared_ptr<LogicalStore> target,
                  std::shared_ptr<LogicalStore> source,
                  std::optional<int32_t> redop);
  void issue_gather(std::shared_ptr<LogicalStore> target,
                    std::shared_ptr<LogicalStore> source,
                    std::shared_ptr<LogicalStore> source_indirect,
                    std::optional<int32_t> redop);
  void issue_scatter(std::shared_ptr<LogicalStore> target,
                     std::shared_ptr<LogicalStore> target_indirect,
                     std::shared_ptr<LogicalStore> source,
                     std::optional<int32_t> redop);
  void issue_scatter_gather(std::shared_ptr<LogicalStore> target,
                            std::shared_ptr<LogicalStore> target_indirect,
                            std::shared_ptr<LogicalStore> source,
                            std::shared_ptr<LogicalStore> source_indirect,
                            std::optional<int32_t> redop);
  void issue_fill(std::shared_ptr<LogicalStore> lhs, std::shared_ptr<LogicalStore> value);
  void tree_reduce(const Library* library,
                   int64_t task_id,
                   std::shared_ptr<LogicalStore> store,
                   std::shared_ptr<LogicalStore> out_store,
                   int64_t radix);
  void flush_scheduling_window();
  void submit(std::unique_ptr<Operation> op);

 public:
  std::shared_ptr<LogicalArray> create_array(std::shared_ptr<Type> type,
                                             uint32_t dim,
                                             bool nullable);
  std::shared_ptr<LogicalArray> create_array(const Shape& extents,
                                             std::shared_ptr<Type> type,
                                             bool nullable,
                                             bool optimize_scalar);
  std::shared_ptr<LogicalArray> create_array_like(std::shared_ptr<LogicalArray> array,
                                                  std::shared_ptr<Type> type);

 private:
  std::shared_ptr<StructLogicalArray> create_struct_array(std::shared_ptr<Type> type,
                                                          uint32_t dim,
                                                          bool nullable);
  std::shared_ptr<StructLogicalArray> create_struct_array(const Shape& extents,
                                                          std::shared_ptr<Type> type,
                                                          bool nullable,
                                                          bool optimize_scalar);

 private:
  std::shared_ptr<BaseLogicalArray> create_base_array(std::shared_ptr<Type> type,
                                                      uint32_t dim,
                                                      bool nullable);
  std::shared_ptr<BaseLogicalArray> create_base_array(const Shape& extents,
                                                      std::shared_ptr<Type> type,
                                                      bool nullable,
                                                      bool optimize_scalar);

 public:
  std::shared_ptr<LogicalStore> create_store(std::shared_ptr<Type> type, uint32_t);
  std::shared_ptr<LogicalStore> create_store(const Shape& extents,
                                             std::shared_ptr<Type> type,
                                             bool optimize_scalar = false);
  [[nodiscard]] std::shared_ptr<LogicalStore> create_store(const Scalar& scalar,
                                                           const Shape& extents);
  std::shared_ptr<LogicalStore> create_store(const Shape& extents,
                                             std::shared_ptr<Type> type,
                                             void* buffer,
                                             bool share,
                                             const mapping::detail::DimOrdering* ordering);

 private:
  void check_dimensionality(uint32_t dim);

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
  Legion::PhysicalRegion map_region_field(Legion::LogicalRegion region, Legion::FieldID field_id);
  void remap_physical_region(Legion::PhysicalRegion pr);
  void unmap_physical_region(Legion::PhysicalRegion pr);
  Legion::Future detach(Legion::PhysicalRegion pr, bool flush, bool unordered);
  uint32_t field_reuse_freq() const;
  bool consensus_match_required() const;
  void progress_unordered_operations() const;

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
  Legion::IndexPartition create_image_partition(const Legion::IndexSpace& index_space,
                                                const Legion::IndexSpace& color_space,
                                                const Legion::LogicalRegion& func_region,
                                                const Legion::LogicalPartition& func_partition,
                                                Legion::FieldID func_field_id,
                                                bool is_range);
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
  template <class T>
  T get_core_tunable(int64_t tunable_id)
  {
    return get_tunable<T>(core_library_->get_mapper_id(), tunable_id);
  }

 public:
  Legion::Future dispatch(Legion::TaskLauncher& launcher,
                          std::vector<Legion::OutputRequirement>& output_requirements);
  Legion::FutureMap dispatch(Legion::IndexTaskLauncher& launcher,
                             std::vector<Legion::OutputRequirement>& output_requirements);
  void dispatch(Legion::CopyLauncher& launcher);
  void dispatch(Legion::IndexCopyLauncher& launcher);
  void dispatch(Legion::FillLauncher& launcher);
  void dispatch(Legion::IndexFillLauncher& launcher);

 public:
  Legion::Future extract_scalar(const Legion::Future& result, uint32_t idx) const;
  Legion::FutureMap extract_scalar(const Legion::FutureMap& result,
                                   uint32_t idx,
                                   const Legion::Domain& launch_domain) const;
  Legion::Future reduce_future_map(const Legion::FutureMap& future_map,
                                   int32_t reduction_op,
                                   const Legion::Future& init_value = Legion::Future()) const;
  Legion::Future reduce_exception_future_map(const Legion::FutureMap& future_map) const;

 public:
  void issue_execution_fence(bool block = false);
  // NOTE: If the type T contains any padding bits, make sure the entries *in the vector* are
  // deterministically zero'd out on all shards, e.g. by doing the initialization as follows:
  //   struct Fred { bool flag; int number; };
  //   std::vector<Fred> input;
  //   input.emplace_back();
  //   memset(&input.back(), 0, sizeof(Fred));
  //   input.back().flag = true;
  //   input.back().flag = number;
  template <typename T>
  ConsensusMatchResult<T> issue_consensus_match(std::vector<T>&& input);

 public:
  void initialize_toplevel_machine();
  const mapping::detail::Machine& get_machine() const;

 public:
  Legion::ProjectionID get_projection(int32_t src_ndim, const proj::SymbolicPoint& point);
  Legion::ProjectionID get_delinearizing_projection();
  Legion::ShardingID get_sharding(const mapping::detail::Machine& machine,
                                  Legion::ProjectionID proj_id);

 private:
  void schedule(std::vector<std::unique_ptr<Operation>> operations);

 public:
  static Runtime* get_runtime();
  static int32_t start(int32_t argc, char** argv);
  bool initialized() const { return initialized_; }
  void destroy();
  int32_t finish();
  const Library* core_library() const { return core_library_; }

 private:
  bool initialized_{false};
  Legion::Runtime* legion_runtime_{nullptr};
  Legion::Context legion_context_{nullptr};
  Library* core_library_{nullptr};

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
  using ShardingDesc = std::pair<Legion::ProjectionID, mapping::ProcessorRange>;
  int64_t next_sharding_id_{LEGATE_CORE_FIRST_DYNAMIC_FUNCTOR_ID};
  std::map<ShardingDesc, Legion::ShardingID> registered_shardings_{};

 private:
  std::vector<std::unique_ptr<Operation>> operations_;
  size_t window_size_{1};
  uint64_t next_unique_id_{0};

 private:
  using RegionFieldID = std::pair<Legion::LogicalRegion, Legion::FieldID>;
  uint64_t next_store_id_{1};
  uint64_t next_storage_id_{1};
  const uint32_t field_reuse_freq_;
  const bool force_consensus_match_;

 private:
  std::map<std::string, Library*> libraries_{};

 private:
  std::map<std::pair<int32_t, int32_t>, int32_t> reduction_ops_{};

 private:
  uint32_t max_pending_exceptions_;
  std::vector<Legion::Future> pending_exceptions_{};
  std::deque<TaskException> outstanding_exceptions_{};
};

void initialize_core_library();

void initialize_core_library_callback(Legion::Machine,
                                      Legion::Runtime*,
                                      const std::set<Processor>&);

void handle_legate_args(int32_t argc, char** argv);

}  // namespace legate::detail

#include "core/runtime/detail/runtime.inl"
