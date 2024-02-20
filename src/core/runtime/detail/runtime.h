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

#include "core/data/detail/scalar.h"
#include "core/data/detail/shape.h"
#include "core/data/external_allocation.h"
#include "core/data/logical_store.h"
#include "core/mapping/machine.h"
#include "core/operation/detail/operation.h"
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
#include "core/utilities/detail/hash.h"
#include "core/utilities/hash.h"
#include "core/utilities/internal_shared_ptr.h"

#include <list>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace legate::detail {

class AutoTask;
class BaseLogicalArray;
class Library;
struct LogicalArray;
class LogicalRegionField;
class ManualTask;
class StructLogicalArray;

struct Config {
  static bool show_progress_requested;
  static bool use_empty_task;
  static bool synchronize_stream_view;
  static bool log_mapping_decisions;
  static bool log_partitioning_decisions;
  static bool has_socket_mem;
  static std::uint64_t max_field_reuse_size;
  static bool warmup_nccl;
};

class Runtime {
 public:
  Runtime();

  [[nodiscard]] Library* create_library(std::string_view library_name,
                                        const ResourceConfig& config,
                                        std::unique_ptr<mapping::Mapper> mapper,
                                        bool in_callback);
  [[nodiscard]] Library* find_library(std::string_view library_name, bool can_fail) const;
  [[nodiscard]] Library* find_or_create_library(std::string_view library_name,
                                                const ResourceConfig& config,
                                                std::unique_ptr<mapping::Mapper> mapper,
                                                bool* created,
                                                bool in_callback);

  void record_reduction_operator(std::uint32_t type_uid,
                                 std::int32_t op_kind,
                                 std::int32_t legion_op_id);
  [[nodiscard]] std::int32_t find_reduction_operator(std::uint32_t type_uid,
                                                     std::int32_t op_kind) const;

  void initialize(Legion::Context legion_context);

  [[nodiscard]] mapping::detail::Machine slice_machine_for_task(const Library* library,
                                                                std::int64_t task_id);
  [[nodiscard]] InternalSharedPtr<AutoTask> create_task(const Library* library,
                                                        std::int64_t task_id);
  [[nodiscard]] InternalSharedPtr<ManualTask> create_task(const Library* library,
                                                          std::int64_t task_id,
                                                          const Domain& launch_domain);
  void issue_copy(InternalSharedPtr<LogicalStore> target,
                  InternalSharedPtr<LogicalStore> source,
                  std::optional<std::int32_t> redop);
  void issue_gather(InternalSharedPtr<LogicalStore> target,
                    InternalSharedPtr<LogicalStore> source,
                    InternalSharedPtr<LogicalStore> source_indirect,
                    std::optional<std::int32_t> redop);
  void issue_scatter(InternalSharedPtr<LogicalStore> target,
                     InternalSharedPtr<LogicalStore> target_indirect,
                     InternalSharedPtr<LogicalStore> source,
                     std::optional<std::int32_t> redop);
  void issue_scatter_gather(InternalSharedPtr<LogicalStore> target,
                            InternalSharedPtr<LogicalStore> target_indirect,
                            InternalSharedPtr<LogicalStore> source,
                            InternalSharedPtr<LogicalStore> source_indirect,
                            std::optional<std::int32_t> redop);
  void issue_fill(const InternalSharedPtr<LogicalArray>& lhs,
                  InternalSharedPtr<LogicalStore> value);
  void issue_fill(const InternalSharedPtr<LogicalArray>& lhs, Scalar value);
  void issue_fill(InternalSharedPtr<LogicalStore> lhs, InternalSharedPtr<LogicalStore> value);
  void issue_fill(InternalSharedPtr<LogicalStore> lhs, Scalar value);
  void tree_reduce(const Library* library,
                   std::int64_t task_id,
                   InternalSharedPtr<LogicalStore> store,
                   InternalSharedPtr<LogicalStore> out_store,
                   std::int32_t radix);
  void flush_scheduling_window();
  void submit(InternalSharedPtr<Operation> op);

  [[nodiscard]] InternalSharedPtr<LogicalArray> create_array(const InternalSharedPtr<Shape>& shape,
                                                             InternalSharedPtr<Type> type,
                                                             bool nullable,
                                                             bool optimize_scalar);
  [[nodiscard]] InternalSharedPtr<LogicalArray> create_array_like(
    const InternalSharedPtr<LogicalArray>& array, InternalSharedPtr<Type> type);
  [[nodiscard]] InternalSharedPtr<LogicalArray> create_list_array(
    InternalSharedPtr<Type> type,
    const InternalSharedPtr<LogicalArray>& descriptor,
    InternalSharedPtr<LogicalArray> vardata);

 private:
  [[nodiscard]] InternalSharedPtr<StructLogicalArray> create_struct_array(
    const InternalSharedPtr<Shape>& shape,
    InternalSharedPtr<Type> type,
    bool nullable,
    bool optimize_scalar);

  [[nodiscard]] InternalSharedPtr<BaseLogicalArray> create_base_array(
    InternalSharedPtr<Shape> shape,
    InternalSharedPtr<Type> type,
    bool nullable,
    bool optimize_scalar);

 public:
  [[nodiscard]] InternalSharedPtr<LogicalStore> create_store(InternalSharedPtr<Type> type,
                                                             std::uint32_t dim);
  [[nodiscard]] InternalSharedPtr<LogicalStore> create_store(InternalSharedPtr<Shape> shape,
                                                             InternalSharedPtr<Type> type,
                                                             bool optimize_scalar);
  [[nodiscard]] InternalSharedPtr<LogicalStore> create_store(const Scalar& scalar,
                                                             InternalSharedPtr<Shape> shape);
  [[nodiscard]] InternalSharedPtr<LogicalStore> create_store(
    const InternalSharedPtr<Shape>& shape,
    InternalSharedPtr<Type> type,
    InternalSharedPtr<ExternalAllocation> allocation,
    const mapping::detail::DimOrdering* ordering);
  using IndexAttachResult =
    std::pair<InternalSharedPtr<LogicalStore>, InternalSharedPtr<LogicalStorePartition>>;
  [[nodiscard]] IndexAttachResult create_store(
    const InternalSharedPtr<Shape>& shape,
    const tuple<std::uint64_t>& tile_shape,
    InternalSharedPtr<Type> type,
    const std::vector<std::pair<legate::ExternalAllocation, tuple<std::uint64_t>>>& allocations,
    const mapping::detail::DimOrdering* ordering);

 private:
  static void check_dimensionality(std::uint32_t dim);
  [[nodiscard]] std::uint64_t current_op_id() const;
  void increment_op_id();

 public:
  void raise_pending_task_exception();
  [[nodiscard]] std::optional<TaskException> check_pending_task_exception();
  void record_pending_exception(const Legion::Future& pending_exception);

  [[nodiscard]] std::uint64_t get_unique_store_id();
  [[nodiscard]] std::uint64_t get_unique_storage_id();

  [[nodiscard]] InternalSharedPtr<LogicalRegionField> create_region_field(
    const InternalSharedPtr<Shape>& shape, std::uint32_t field_size);
  [[nodiscard]] InternalSharedPtr<LogicalRegionField> import_region_field(
    const InternalSharedPtr<Shape>& shape,
    Legion::LogicalRegion region,
    Legion::FieldID field_id,
    std::uint32_t field_size);
  [[nodiscard]] Legion::PhysicalRegion map_region_field(Legion::LogicalRegion region,
                                                        Legion::FieldID field_id);
  void remap_physical_region(Legion::PhysicalRegion pr);
  void unmap_physical_region(Legion::PhysicalRegion pr);
  [[nodiscard]] Legion::Future detach(const Legion::PhysicalRegion& physical_region,
                                      bool flush,
                                      bool unordered);
  [[nodiscard]] Legion::Future detach(const Legion::ExternalResources& external_resources,
                                      bool flush,
                                      bool unordered);
  [[nodiscard]] std::uint32_t field_reuse_freq() const;
  [[nodiscard]] bool consensus_match_required() const;
  void progress_unordered_operations() const;

  [[nodiscard]] RegionManager* find_or_create_region_manager(const Legion::IndexSpace& index_space);
  [[nodiscard]] FieldManager* find_or_create_field_manager(InternalSharedPtr<Shape> shape,
                                                           std::uint32_t field_size);
  [[nodiscard]] CommunicatorManager* communicator_manager() const;
  [[nodiscard]] MachineManager* machine_manager() const;
  [[nodiscard]] PartitionManager* partition_manager() const;
  [[nodiscard]] ProvenanceManager* provenance_manager() const;

  [[nodiscard]] const Legion::IndexSpace& find_or_create_index_space(
    const tuple<std::uint64_t>& extents);
  [[nodiscard]] const Legion::IndexSpace& find_or_create_index_space(const Domain& domain);
  [[nodiscard]] Legion::IndexPartition create_restricted_partition(
    const Legion::IndexSpace& index_space,
    const Legion::IndexSpace& color_space,
    Legion::PartitionKind kind,
    const Legion::DomainTransform& transform,
    const Legion::Domain& extent);
  [[nodiscard]] Legion::IndexPartition create_weighted_partition(
    const Legion::IndexSpace& index_space,
    const Legion::IndexSpace& color_space,
    const Legion::FutureMap& weights);
  [[nodiscard]] Legion::IndexPartition create_image_partition(
    const Legion::IndexSpace& index_space,
    const Legion::IndexSpace& color_space,
    const Legion::LogicalRegion& func_region,
    const Legion::LogicalPartition& func_partition,
    Legion::FieldID func_field_id,
    bool is_range,
    const mapping::detail::Machine& machine);
  [[nodiscard]] Legion::FieldSpace create_field_space();
  [[nodiscard]] Legion::LogicalRegion create_region(const Legion::IndexSpace& index_space,
                                                    const Legion::FieldSpace& field_space);
  void destroy_region(const Legion::LogicalRegion& logical_region, bool unordered = false);
  [[nodiscard]] Legion::LogicalPartition create_logical_partition(
    const Legion::LogicalRegion& logical_region, const Legion::IndexPartition& index_partition);
  [[nodiscard]] Legion::LogicalRegion get_subregion(const Legion::LogicalPartition& partition,
                                                    const Legion::DomainPoint& color);
  [[nodiscard]] Legion::LogicalRegion find_parent_region(const Legion::LogicalRegion& region);
  [[nodiscard]] Legion::FieldID allocate_field(const Legion::FieldSpace& field_space,
                                               std::size_t field_size);
  [[nodiscard]] Legion::FieldID allocate_field(const Legion::FieldSpace& field_space,
                                               Legion::FieldID field_id,
                                               std::size_t field_size);
  [[nodiscard]] Legion::Domain get_index_space_domain(const Legion::IndexSpace& index_space) const;
  [[nodiscard]] Legion::FutureMap delinearize_future_map(
    const Legion::FutureMap& future_map, const Legion::IndexSpace& new_domain) const;

  [[nodiscard]] std::pair<Legion::PhaseBarrier, Legion::PhaseBarrier> create_barriers(
    std::size_t num_tasks);
  void destroy_barrier(Legion::PhaseBarrier barrier);

  [[nodiscard]] Legion::Future get_tunable(Legion::MapperID mapper_id,
                                           std::int64_t tunable_id,
                                           std::size_t size);

  template <class T>
  [[nodiscard]] T get_tunable(Legion::MapperID mapper_id, std::int64_t tunable_id);
  template <class T>
  [[nodiscard]] T get_core_tunable(std::int64_t tunable_id);

  [[nodiscard]] Legion::Future dispatch(
    Legion::TaskLauncher& launcher, std::vector<Legion::OutputRequirement>& output_requirements);
  [[nodiscard]] Legion::FutureMap dispatch(
    Legion::IndexTaskLauncher& launcher,
    std::vector<Legion::OutputRequirement>& output_requirements);

  void dispatch(Legion::CopyLauncher& launcher);
  void dispatch(Legion::IndexCopyLauncher& launcher);
  void dispatch(Legion::FillLauncher& launcher);
  void dispatch(Legion::IndexFillLauncher& launcher);

  [[nodiscard]] Legion::Future extract_scalar(const Legion::Future& result,
                                              std::uint32_t idx) const;
  [[nodiscard]] Legion::FutureMap extract_scalar(const Legion::FutureMap& result,
                                                 std::uint32_t idx,
                                                 const Legion::Domain& launch_domain) const;
  [[nodiscard]] Legion::Future reduce_future_map(
    const Legion::FutureMap& future_map,
    std::int32_t reduction_op,
    const Legion::Future& init_value = Legion::Future()) const;
  [[nodiscard]] Legion::Future reduce_exception_future_map(
    const Legion::FutureMap& future_map) const;

  void discard_field(const Legion::LogicalRegion& region, Legion::FieldID field_id);
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
  [[nodiscard]] ConsensusMatchResult<T> issue_consensus_match(std::vector<T>&& input);

  void initialize_toplevel_machine();
  [[nodiscard]] const mapping::detail::Machine& get_machine() const;
  [[nodiscard]] const mapping::detail::LocalMachine& local_machine() const;

  [[nodiscard]] Legion::ProjectionID get_affine_projection(std::uint32_t src_ndim,
                                                           const proj::SymbolicPoint& point);
  [[nodiscard]] Legion::ProjectionID get_delinearizing_projection(
    const tuple<std::uint64_t>& color_shape);
  [[nodiscard]] Legion::ProjectionID get_compound_projection(
    const tuple<std::uint64_t>& color_shape, const proj::SymbolicPoint& point);
  [[nodiscard]] Legion::ShardingID get_sharding(const mapping::detail::Machine& machine,
                                                Legion::ProjectionID proj_id);

 private:
  static void schedule(const std::vector<InternalSharedPtr<Operation>>& operations);

 public:
  [[nodiscard]] static Runtime* get_runtime();
  [[nodiscard]] static std::int32_t start(std::int32_t argc, char** argv);
  [[nodiscard]] bool initialized() const;
  void register_shutdown_callback(ShutdownCallback callback);
  void destroy();
  [[nodiscard]] std::int32_t finish();
  [[nodiscard]] const Library* core_library() const;

 private:
  bool initialized_{};
  Legion::Runtime* legion_runtime_{};
  Legion::Context legion_context_{};
  Library* core_library_{};
  std::list<ShutdownCallback> callbacks_{};
  legate::mapping::detail::LocalMachine local_machine_{};

  using FieldManagerKey = std::pair<Legion::IndexSpace, std::uint32_t>;
  std::unordered_map<FieldManagerKey, std::unique_ptr<FieldManager>, hasher<FieldManagerKey>>
    field_managers_{};
  using RegionManagerKey = Legion::IndexSpace;
  std::unordered_map<RegionManagerKey, std::unique_ptr<RegionManager>> region_managers_{};
  std::unique_ptr<CommunicatorManager> communicator_manager_{};
  std::unique_ptr<MachineManager> machine_manager_{};
  std::unique_ptr<PartitionManager> partition_manager_{};
  std::unique_ptr<ProvenanceManager> provenance_manager_{};

  std::unordered_map<Domain, Legion::IndexSpace> cached_index_spaces_{};

  using AffineProjectionDesc   = std::pair<uint32_t, proj::SymbolicPoint>;
  using CompoundProjectionDesc = std::pair<tuple<std::uint64_t>, proj::SymbolicPoint>;
  std::int64_t next_projection_id_{LEGATE_CORE_FIRST_DYNAMIC_FUNCTOR_ID};
  std::unordered_map<AffineProjectionDesc, Legion::ProjectionID, hasher<AffineProjectionDesc>>
    affine_projections_{};
  std::unordered_map<tuple<std::uint64_t>, Legion::ProjectionID, hasher<tuple<std::uint64_t>>>
    delinearizing_projections_{};
  std::unordered_map<CompoundProjectionDesc, Legion::ProjectionID, hasher<CompoundProjectionDesc>>
    compound_projections_{};

  using ShardingDesc = std::pair<Legion::ProjectionID, mapping::ProcessorRange>;
  std::int64_t next_sharding_id_{LEGATE_CORE_FIRST_DYNAMIC_FUNCTOR_ID};
  std::unordered_map<ShardingDesc, Legion::ShardingID, hasher<ShardingDesc>>
    registered_shardings_{};

  std::vector<InternalSharedPtr<Operation>> operations_;
  std::size_t window_size_{1};
  std::uint64_t current_op_id_{};

  using RegionFieldID = std::pair<Legion::LogicalRegion, Legion::FieldID>;
  std::uint64_t next_store_id_{1};
  std::uint64_t next_storage_id_{1};
  std::uint32_t field_reuse_freq_{};
  bool force_consensus_match_{};

  // This could be a hash map, but kept as an ordered map just in case we may later support
  // library-specific shutdown callbacks that can launch tasks.
  std::map<std::string, std::unique_ptr<Library>, std::less<>> libraries_{};

  using ReductionOpTableKey = std::pair<uint32_t, std::int32_t>;
  std::unordered_map<ReductionOpTableKey, int32_t, hasher<ReductionOpTableKey>> reduction_ops_{};

  // TODO(wonchanl): We keep some of the deferred exception code as we will put it back later
  std::vector<Legion::Future> pending_exceptions_{};
  std::deque<TaskException> outstanding_exceptions_{};
};

void initialize_core_library();

void initialize_core_library_callback(Legion::Machine,
                                      Legion::Runtime*,
                                      const std::set<Processor>&);

void handle_legate_args(std::int32_t argc, char** argv);

}  // namespace legate::detail

#include "core/runtime/detail/runtime.inl"
