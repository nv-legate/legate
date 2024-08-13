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

#pragma once

#include "core/data/detail/scalar.h"
#include "core/data/detail/shape.h"
#include "core/data/external_allocation.h"
#include "core/data/logical_store.h"
#include "core/mapping/detail/instance_manager.h"
#include "core/mapping/machine.h"
#include "core/operation/detail/operation.h"
#include "core/runtime/detail/communicator_manager.h"
#include "core/runtime/detail/consensus_match_result.h"
#include "core/runtime/detail/field_manager.h"
#include "core/runtime/detail/library.h"
#include "core/runtime/detail/partition_manager.h"
#include "core/runtime/detail/projection.h"
#include "core/runtime/detail/region_manager.h"
#include "core/runtime/detail/scope.h"
#include "core/runtime/resource.h"
#include "core/task/detail/returned_exception.h"
#include "core/type/type_info.h"
#include "core/utilities/detail/core_ids.h"
#include "core/utilities/detail/hash.h"
#include "core/utilities/detail/zstring_view.h"
#include "core/utilities/hash.h"
#include "core/utilities/internal_shared_ptr.h"

#include <list>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

struct CUstream_st;

namespace legate::detail {

class AutoTask;
class BaseLogicalArray;
class Library;
class LogicalArray;
class LogicalRegionField;
class ManualTask;
class StructLogicalArray;

class Runtime {
 public:
  Runtime();

  [[nodiscard]] Library* create_library(std::string_view library_name,
                                        const ResourceConfig& config,
                                        std::unique_ptr<mapping::Mapper> mapper,
                                        std::map<VariantCode, VariantOptions> default_options,
                                        bool in_callback);
  [[nodiscard]] const Library* find_library(std::string_view library_name, bool can_fail) const;
  [[nodiscard]] Library* find_library(std::string_view library_name, bool can_fail);
  [[nodiscard]] Library* find_or_create_library(
    std::string_view library_name,
    const ResourceConfig& config,
    std::unique_ptr<mapping::Mapper> mapper,
    const std::map<VariantCode, VariantOptions>& default_options,
    bool* created,
    bool in_callback);

  void record_reduction_operator(std::uint32_t type_uid,
                                 std::int32_t op_kind,
                                 GlobalRedopID legion_op_id);
  [[nodiscard]] GlobalRedopID find_reduction_operator(std::uint32_t type_uid,
                                                      std::int32_t op_kind) const;

  void initialize(Legion::Context legion_context, std::int32_t argc, char** argv);

  [[nodiscard]] mapping::detail::Machine slice_machine_for_task(const Library* library,
                                                                LocalTaskID task_id) const;
  [[nodiscard]] InternalSharedPtr<AutoTask> create_task(const Library* library,
                                                        LocalTaskID task_id);
  [[nodiscard]] InternalSharedPtr<ManualTask> create_task(const Library* library,
                                                          LocalTaskID task_id,
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
                   LocalTaskID task_id,
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
  [[nodiscard]] InternalSharedPtr<StructLogicalArray> create_struct_array_(
    const InternalSharedPtr<Shape>& shape,
    InternalSharedPtr<Type> type,
    bool nullable,
    bool optimize_scalar);

  [[nodiscard]] InternalSharedPtr<BaseLogicalArray> create_base_array_(
    InternalSharedPtr<Shape> shape,
    InternalSharedPtr<Type> type,
    bool nullable,
    bool optimize_scalar);

 public:
  [[nodiscard]] InternalSharedPtr<LogicalStore> create_store(InternalSharedPtr<Type> type,
                                                             std::uint32_t dim,
                                                             bool optimize_scalar = false);
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
  static void check_dimensionality_(std::uint32_t dim);
  [[nodiscard]] std::uint64_t current_op_id_() const;
  void increment_op_id_();

 public:
  void raise_pending_exception();
  [[nodiscard]] std::optional<ReturnedException> check_pending_task_exception();
  void record_pending_exception(Legion::Future pending_exception);

  [[nodiscard]] std::uint64_t get_unique_store_id();
  [[nodiscard]] std::uint64_t get_unique_storage_id();

  [[nodiscard]] InternalSharedPtr<LogicalRegionField> create_region_field(
    InternalSharedPtr<Shape> shape, std::uint32_t field_size);
  [[nodiscard]] InternalSharedPtr<LogicalRegionField> import_region_field(
    InternalSharedPtr<Shape> shape,
    Legion::LogicalRegion region,
    Legion::FieldID field_id,
    std::uint32_t field_size);
  void attach_alloc_info(const InternalSharedPtr<LogicalRegionField>& rf,
                         std::string_view provenance);
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
  [[nodiscard]] std::size_t field_reuse_size() const;
  [[nodiscard]] bool consensus_match_required() const;
  void progress_unordered_operations() const;

  [[nodiscard]] RegionManager* find_or_create_region_manager(const Legion::IndexSpace& index_space);
  [[nodiscard]] FieldManager* field_manager();
  [[nodiscard]] CommunicatorManager* communicator_manager();
  [[nodiscard]] const CommunicatorManager* communicator_manager() const;
  [[nodiscard]] PartitionManager* partition_manager();
  [[nodiscard]] const PartitionManager* partition_manager() const;
  [[nodiscard]] Scope& scope();
  [[nodiscard]] const Scope& scope() const;

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
  [[nodiscard]] Legion::IndexPartition create_approximate_image_partition(
    const InternalSharedPtr<LogicalStore>& store,
    const InternalSharedPtr<Partition>& partition,
    const Legion::IndexSpace& index_space,
    bool sorted);
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
                                               Legion::FieldID field_id,
                                               std::size_t field_size);
  [[nodiscard]] Legion::Domain get_index_space_domain(const Legion::IndexSpace& index_space) const;
  [[nodiscard]] Legion::FutureMap delinearize_future_map(const Legion::FutureMap& future_map,
                                                         const Domain& new_domain);
  [[nodiscard]] Legion::FutureMap reshape_future_map(const Legion::FutureMap& future_map,
                                                     const Domain& new_domain);

  [[nodiscard]] std::pair<Legion::PhaseBarrier, Legion::PhaseBarrier> create_barriers(
    std::size_t num_tasks);
  void destroy_barrier(Legion::PhaseBarrier barrier);

  [[nodiscard]] Legion::Future get_tunable(Legion::MapperID mapper_id, std::int64_t tunable_id);

  [[nodiscard]] Legion::Future dispatch(
    Legion::TaskLauncher& launcher, std::vector<Legion::OutputRequirement>& output_requirements);
  [[nodiscard]] Legion::FutureMap dispatch(
    Legion::IndexTaskLauncher& launcher,
    std::vector<Legion::OutputRequirement>& output_requirements);

  void dispatch(const Legion::CopyLauncher& launcher);
  void dispatch(const Legion::IndexCopyLauncher& launcher);
  void dispatch(const Legion::FillLauncher& launcher);
  void dispatch(const Legion::IndexFillLauncher& launcher);

  [[nodiscard]] Legion::Future extract_scalar(const Legion::Future& result,
                                              std::size_t offset,
                                              std::size_t size) const;
  [[nodiscard]] Legion::FutureMap extract_scalar(const Legion::FutureMap& result,
                                                 std::size_t offset,
                                                 std::size_t size,
                                                 const Legion::Domain& launch_domain) const;
  [[nodiscard]] Legion::Future reduce_future_map(
    const Legion::FutureMap& future_map,
    GlobalRedopID reduction_op,
    const Legion::Future& init_value = Legion::Future{}) const;
  [[nodiscard]] Legion::Future reduce_exception_future_map(
    const Legion::FutureMap& future_map) const;

  void discard_field(const Legion::LogicalRegion& region, Legion::FieldID field_id);
  void issue_mapping_fence();
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

  void begin_trace(std::uint32_t trace_id);
  void end_trace(std::uint32_t trace_id);

  InternalSharedPtr<mapping::detail::Machine> create_toplevel_machine();
  [[nodiscard]] const mapping::detail::Machine& get_machine() const;
  [[nodiscard]] ZStringView get_provenance() const;
  [[nodiscard]] const mapping::detail::LocalMachine& local_machine() const;
  [[nodiscard]] std::uint32_t node_count() const;
  [[nodiscard]] std::uint32_t node_id() const;

  [[nodiscard]] Legion::ProjectionID get_affine_projection(std::uint32_t src_ndim,
                                                           const proj::SymbolicPoint& point);
  [[nodiscard]] Legion::ProjectionID get_delinearizing_projection(
    const tuple<std::uint64_t>& color_shape);
  [[nodiscard]] Legion::ProjectionID get_compound_projection(
    const tuple<std::uint64_t>& color_shape, const proj::SymbolicPoint& point);
  [[nodiscard]] Legion::ShardingID get_sharding(const mapping::detail::Machine& machine,
                                                Legion::ProjectionID proj_id);

  [[nodiscard]] Processor get_executing_processor() const;
  [[nodiscard]] const InternalSharedPtr<mapping::detail::InstanceManager>& get_instance_manager()
    const;
  [[nodiscard]] const InternalSharedPtr<mapping::detail::ReductionInstanceManager>&
  get_reduction_instance_manager() const;

 private:
  static void schedule_(std::vector<InternalSharedPtr<Operation>>&& operations);

 public:
  [[nodiscard]] static Runtime* get_runtime();
  [[nodiscard]] static std::int32_t start(std::int32_t argc, char** argv);
  [[nodiscard]] bool initialized() const;
  void register_shutdown_callback(ShutdownCallback callback);
  [[nodiscard]] std::int32_t finish();
  [[nodiscard]] const Library* core_library() const;

  [[nodiscard]] CUstream_st* get_cuda_stream() const;

 private:
  bool initialized_{};
  Legion::Runtime* legion_runtime_{};
  Legion::Context legion_context_{};
  Library* core_library_{};
  std::list<ShutdownCallback> callbacks_{};
  legate::mapping::detail::LocalMachine local_machine_{};

  std::unique_ptr<FieldManager> field_manager_{};
  using RegionManagerKey = Legion::IndexSpace;
  std::unordered_map<RegionManagerKey, RegionManager> region_managers_{};
  std::optional<CommunicatorManager> communicator_manager_{};
  std::optional<PartitionManager> partition_manager_{};
  Scope scope_{};

  std::unordered_map<Domain, Legion::IndexSpace> cached_index_spaces_{};

  using AffineProjectionDesc   = std::pair<uint32_t, proj::SymbolicPoint>;
  using CompoundProjectionDesc = std::pair<tuple<std::uint64_t>, proj::SymbolicPoint>;
  std::int64_t next_projection_id_{
    static_cast<std::int64_t>(CoreProjectionOp::FIRST_DYNAMIC_FUNCTOR)};
  std::unordered_map<AffineProjectionDesc, Legion::ProjectionID, hasher<AffineProjectionDesc>>
    affine_projections_{};
  std::unordered_map<tuple<std::uint64_t>, Legion::ProjectionID, hasher<tuple<std::uint64_t>>>
    delinearizing_projections_{};
  std::unordered_map<CompoundProjectionDesc, Legion::ProjectionID, hasher<CompoundProjectionDesc>>
    compound_projections_{};

  using ShardingDesc = std::pair<Legion::ProjectionID, mapping::ProcessorRange>;
  std::int64_t next_sharding_id_{
    static_cast<std::int64_t>(CoreProjectionOp::FIRST_DYNAMIC_FUNCTOR)};
  std::unordered_map<ShardingDesc, Legion::ShardingID, hasher<ShardingDesc>>
    registered_shardings_{};

  std::vector<InternalSharedPtr<Operation>> operations_{};
  std::size_t window_size_{1};
  std::uint64_t cur_op_id_{};

  using RegionFieldID = std::pair<Legion::LogicalRegion, Legion::FieldID>;
  std::uint64_t next_store_id_{1};
  std::uint64_t next_storage_id_{1};
  std::uint32_t field_reuse_freq_{};
  std::size_t field_reuse_size_{1};
  bool force_consensus_match_{};

  // This could be a hash map, but kept as an ordered map just in case we may later support
  // library-specific shutdown callbacks that can launch tasks.
  std::map<std::string, Library, std::less<>> libraries_{};

  using ReductionOpTableKey = std::pair<std::uint32_t, std::int32_t>;
  std::unordered_map<ReductionOpTableKey, GlobalRedopID, hasher<ReductionOpTableKey>>
    reduction_ops_{};

  std::vector<Legion::Future> pending_exceptions_{};

  InternalSharedPtr<mapping::detail::InstanceManager> instance_manager_{};
  InternalSharedPtr<mapping::detail::ReductionInstanceManager> reduction_instance_manager_{};
};

[[nodiscard]] bool has_started();
[[nodiscard]] bool has_finished();

}  // namespace legate::detail

#include "core/runtime/detail/runtime.inl"
