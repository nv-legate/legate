/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/cuda/detail/cuda_driver_types.h>
#include <legate/cuda/detail/module_manager.h>
#include <legate/data/detail/logical_region_field.h>
#include <legate/data/detail/scalar.h>
#include <legate/mapping/detail/mapping.h>
#include <legate/mapping/machine.h>
#include <legate/operation/detail/timing.h>
#include <legate/runtime/detail/communicator_manager.h>
#include <legate/runtime/detail/config.h>
#include <legate/runtime/detail/consensus_match_result.h>
#include <legate/runtime/detail/library.h>
#include <legate/runtime/detail/mapper_manager.h>
#include <legate/runtime/detail/partition_manager.h>
#include <legate/runtime/detail/projection.h>
#include <legate/runtime/detail/region_manager.h>
#include <legate/runtime/detail/scope.h>
#include <legate/task/detail/returned_exception.h>
#include <legate/task/variant_options.h>
#include <legate/type/types.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/detail/hash.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/detail/zstring_view.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/typedefs.h>

#include <atomic>
#include <deque>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace Legion {  // NOLINT

struct RegistrationCallbackArgs;

}  // namespace Legion

namespace legate {

class TaskInfo;
class VariantInfo;
struct ResourceConfig;
class ExternalAllocation;
class ParallelPolicy;

}  // namespace legate

namespace legate::mapping {

class Mapper;

}  // namespace legate::mapping

namespace legate::detail {

class AutoTask;
class BaseLogicalArray;
class LogicalArray;
class ManualTask;
class PhysicalTask;
class StructLogicalArray;
class LogicalStore;
class LogicalStorePartition;
class Operation;
class FieldManager;
class Shape;

class Runtime {
 public:
  /**
   * @brief Construct a Runtime instance.
   *
   * @param config The configuration for the runtime.
   */
  explicit Runtime(Config config);
  // The runtime is a singleton and is not copyable in any way
  Runtime(const Runtime&)            = delete;
  Runtime& operator=(const Runtime&) = delete;
  Runtime(Runtime&&)                 = delete;
  Runtime& operator=(Runtime&&)      = delete;

  /**
   * @return The configuration object for the runtime.
   */
  [[nodiscard]] const Config& config() const;

  [[nodiscard]] Library& create_library(std::string_view library_name,
                                        const ResourceConfig& config,
                                        std::unique_ptr<mapping::Mapper> mapper,
                                        std::map<VariantCode, VariantOptions> default_options);
  [[nodiscard]] std::optional<std::reference_wrapper<const Library>> find_library(
    std::string_view library_name) const;
  [[nodiscard]] std::optional<std::reference_wrapper<Library>> find_library(
    std::string_view library_name);
  [[nodiscard]] Library& find_or_create_library(
    std::string_view library_name,
    const ResourceConfig& config,
    std::unique_ptr<mapping::Mapper> mapper,
    const std::map<VariantCode, VariantOptions>& default_options,
    bool* created);

  void record_reduction_operator(std::uint32_t type_uid,
                                 std::int32_t op_kind,
                                 GlobalRedopID legion_op_id);
  [[nodiscard]] GlobalRedopID find_reduction_operator(std::uint32_t type_uid,
                                                      std::int32_t op_kind) const;

  [[nodiscard]] InternalSharedPtr<AutoTask> create_task(const Library& library,
                                                        LocalTaskID task_id);
  [[nodiscard]] InternalSharedPtr<ManualTask> create_task(const Library& library,
                                                          LocalTaskID task_id,
                                                          const Domain& launch_domain);
  /**
   * @brief Creates a PhysicalTask using top-level machine context.
   *
   * This method creates a PhysicalTask that will execute on the machine determined by the
   * runtime's top-level context. This is appropriate for tasks launched from the main
   * execution context, not from within other tasks.
   *
   * PhysicalTasks are designed for direct, inline execution with pre-partitioned physical
   * arrays. They bypass Legate's automatic partitioning and are executed immediately
   * without going through Legion's task scheduling.
   *
   * @param library The library containing the task implementation
   * @param task_id Local task identifier within the library
   * @return A shared pointer to the created PhysicalTask
   */
  [[nodiscard]] InternalSharedPtr<PhysicalTask> create_physical_task(const Library& library,
                                                                     LocalTaskID task_id);

  /**
   * @brief Creates a PhysicalTask using TaskContext machine for correct allocation in nested tasks.
   *
   * This method creates a PhysicalTask that will execute on the machine associated with the
   * provided TaskContext. This is the correct method to use when creating PhysicalTasks from
   * within other tasks (nested execution), as it ensures the new task uses the appropriate
   * machine context from the parent task.
   *
   * @param context The TaskContext from the parent task, providing machine and execution context
   * @param library The library containing the task implementation
   * @param task_id Local task identifier within the library
   * @return A shared pointer to the created PhysicalTask
   */
  [[nodiscard]] InternalSharedPtr<PhysicalTask> create_physical_task(
    const legate::TaskContext& context, const Library& library, LocalTaskID task_id);
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
  void tree_reduce(const Library& library,
                   LocalTaskID task_id,
                   InternalSharedPtr<LogicalStore> store,
                   InternalSharedPtr<LogicalStore> out_store,
                   std::int32_t radix);

  // also used for offloading stores
  void offload_to(mapping::StoreTarget target_mem, const InternalSharedPtr<LogicalArray>& array);

  /**
   * @brief Launch operations in Legate's queue.
   *
   * @param streaming_scope_change set to true when flush_scheduling_window is
   * called due to entering or leaving a streaming scope.
   *
   * @throws std::invalid_argument if called during or at the end of a strict
   * streaming scope.
   */
  void flush_scheduling_window(bool streaming_scope_change = false);
  void submit(InternalSharedPtr<Operation> op);
  static void launch_immediately(const InternalSharedPtr<Operation>& op);

  [[nodiscard]] InternalSharedPtr<LogicalArray> create_array(const InternalSharedPtr<Shape>& shape,
                                                             InternalSharedPtr<Type> type,
                                                             bool nullable,
                                                             bool optimize_scalar);
  [[nodiscard]] InternalSharedPtr<LogicalArray> create_array_like(
    const InternalSharedPtr<LogicalArray>& array, InternalSharedPtr<Type> type);
  /**
   * @brief Creates a nullable array from a given store and null mask.
   *
   * @param store Store for the array's data.
   * @param null_mask Store for the array's null mask.
   *
   * @note This call can block if either `store` or `null_mask` is unbound.
   *
   * @return Nullable logical array.
   *
   * @throw std::invalid_argument When any of the following is true:
   * #. `null_mask` is not of boolean type.
   * #. `store` and `null_mask` have different shapes.
   * #. `store` or `null_mask` are not top-level stores.
   */
  [[nodiscard]] InternalSharedPtr<LogicalArray> create_nullable_array(
    const InternalSharedPtr<LogicalStore>& store, const InternalSharedPtr<LogicalStore>& null_mask);
  [[nodiscard]] InternalSharedPtr<LogicalArray> create_list_array(
    InternalSharedPtr<Type> type,
    const InternalSharedPtr<LogicalArray>& descriptor,
    InternalSharedPtr<LogicalArray> vardata);
  /**
   * @brief Creates a struct array from existing sub-arrays and null mask.
   *
   * The caller is responsible for making sure that the fields sub-arrays are valid.
   *
   * @param fields Sub-arrays for fields.
   * @param null_mask Optional null mask for the struct array.
   *
   * @note This call can block if either `fields` or `null_mask` is unbound.
   *
   * @return Struct logical array
   *
   * @throw std::invalid_argument When any of the following is true:
   * #. `null_mask` is not of boolean type if provided.
   * #.  any of `fields` or `null_mask`, if provided, have different shapes.
   * #.  any of the `fields` or `null_mask` are transformed (i.e., not a top-level store).
   */
  [[nodiscard]] InternalSharedPtr<StructLogicalArray> create_struct_array(
    SmallVector<InternalSharedPtr<LogicalArray>>&& fields,
    const std::optional<InternalSharedPtr<LogicalStore>>& null_mask);

  /**
   * @brief Give access to certain methods via this class.
   */
  class PrivateKey {
    PrivateKey() = default;
    friend class legate::detail::Scope;
  };

  /**
   * @brief something went wrong, such as an exception or error inside a streaming
   * scope. So clear the tasks in the queue.
   */
  void clear_scheduling_window(PrivateKey);

 private:
  /**
   * @brief Launch the operations in the queue provided.
   *
   * @param window queue of tasks.
   */
  void schedule_(std::deque<InternalSharedPtr<Operation>>* window);

  [[nodiscard]] std::pair<mapping::detail::Machine, const VariantInfo&> slice_machine_for_task_(
    const TaskInfo& info) const;

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
    InternalSharedPtr<mapping::detail::DimOrdering> ordering);
  using IndexAttachResult =
    std::pair<InternalSharedPtr<LogicalStore>, InternalSharedPtr<LogicalStorePartition>>;
  [[nodiscard]] IndexAttachResult create_store(
    const InternalSharedPtr<Shape>& shape,
    const SmallVector<std::uint64_t, LEGATE_MAX_DIM>& tile_shape,
    InternalSharedPtr<Type> type,
    Span<const std::pair<legate::ExternalAllocation, tuple<std::uint64_t>>> allocations,
    InternalSharedPtr<mapping::detail::DimOrdering> ordering);

  void prefetch_bloated_instances(InternalSharedPtr<LogicalStore> store,
                                  SmallVector<std::uint64_t, LEGATE_MAX_DIM> low_offsets,
                                  SmallVector<std::uint64_t, LEGATE_MAX_DIM> high_offsets,
                                  bool initialize);

 private:
  static void check_dimensionality_(std::uint32_t dim);

 public:
  /**
   * @return A new unique operation ID.
   */
  [[nodiscard]] std::uint64_t new_op_id();

  void raise_pending_exception();
  [[nodiscard]] std::optional<ReturnedException> check_pending_task_exception();
  void record_pending_exception(Legion::Future pending_exception);
  void record_pending_exception(ReturnedException pending_exception);

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
                                                        Legion::FieldID field_id,
                                                        legate::mapping::StoreTarget target);
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
  void progress_unordered_operations();

  [[nodiscard]] RegionManager& find_or_create_region_manager(const Legion::IndexSpace& index_space);
  [[nodiscard]] FieldManager& field_manager();
  [[nodiscard]] CommunicatorManager& communicator_manager();
  [[nodiscard]] const CommunicatorManager& communicator_manager() const;
  [[nodiscard]] PartitionManager& partition_manager();
  [[nodiscard]] const PartitionManager& partition_manager() const;
  [[nodiscard]] Scope& scope();
  [[nodiscard]] const Scope& scope() const;

  [[nodiscard]] const Legion::IndexSpace& find_or_create_index_space(
    Span<const std::uint64_t> extents);
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
  [[nodiscard]] Legion::Domain get_index_space_domain(const Legion::IndexSpace& index_space);
  [[nodiscard]] Legion::FutureMap delinearize_future_map(const Legion::FutureMap& future_map,
                                                         const Domain& new_domain);
  [[nodiscard]] Legion::FutureMap reshape_future_map(const Legion::FutureMap& future_map,
                                                     const Domain& new_domain);

  [[nodiscard]] std::pair<Legion::PhaseBarrier, Legion::PhaseBarrier> create_barriers(
    std::size_t num_tasks);
  void destroy_barrier(Legion::PhaseBarrier barrier);

  [[nodiscard]] Legion::Future get_tunable(const Library& library, std::int64_t tunable_id);

  [[nodiscard]] Legion::Future dispatch(
    Legion::TaskLauncher& launcher, std::vector<Legion::OutputRequirement>& output_requirements);
  [[nodiscard]] Legion::FutureMap dispatch(
    Legion::IndexTaskLauncher& launcher,
    std::vector<Legion::OutputRequirement>& output_requirements);

  void dispatch(const Legion::CopyLauncher& launcher);
  void dispatch(const Legion::IndexCopyLauncher& launcher);
  void dispatch(const Legion::FillLauncher& launcher);
  void dispatch(const Legion::IndexFillLauncher& launcher);

  /**
   * @brief Given a future (`result`) that contains possibly multiple future values packed into
   * a single value, extract a specific value and return it as a standalone future.
   *
   * @param parallel_policy The policy to use for the task launch.
   * @param result The future to extract from.
   * @param offset The byte-based offset into the buffer of `result` where the sub-value begins.
   * @param size The maximum possible size of the value to extract. The actual value may be smaller.
   * @param future_size The size (in bytes) of buffer held by `result`.
   *
   * `size` is an upper bound. While the actual size of the value may be smaller, it cannot
   * exceed `size`. As `size` would also affect other offsets, the only time where the true
   * size of the future may differ from `size` is if it is the "last" value contained in the future.
   *
   * `future_size` is passed explicitly (instead of using `result.get_untyped_size()`) because
   * getting the size of the future requires blocking on the completion of the task that fills
   * the future. This could end up in a deadlock if the future hasn't been allocated yet by the
   * mapper thread.
   *
   * @see `legate::detail::Task::demux_scalar_stores_()`.
   */
  [[nodiscard]] Legion::Future extract_scalar(const ParallelPolicy& parallel_policy,
                                              const Legion::Future& result,
                                              std::size_t offset,
                                              std::size_t size,
                                              std::size_t future_size) const;
  /**
   * @brief Given a future-map (`result`) that contains possibly multiple future-map values
   * packed into a single one, extract a specific value and return it as a standalone
   * future-map.
   *
   * @param parallel_policy The policy to use for the task launch.
   * @param result The future-map to extract from.
   * @param offset The byte-based offset into the buffer of `result` where the sub-values begin.
   * @param size The maximum possible size of the value to extract. The actual value may be smaller.
   * @param future_size The size (in bytes) of buffer held by `result`.
   * @param launch_domain The launch domain of the task that produced `result`.
   *
   * `size` is an upper bound. While the actual size of the value may be smaller, it cannot
   * exceed `size`. As `size` would also affect other offsets, the only time where the true
   * size of the future may differ from `size` is if it is the "last" value contained in the future.
   *
   * `future_size` is passed explicitly (instead of using `result.get_untyped_size()`) because
   * getting the size of the future requires blocking on the completion of the task that fills
   * the future. This could end up in a deadlock if the future hasn't been allocated yet by the
   * mapper thread.
   *
   * @see `legate::detail::Task::demux_scalar_stores_()`.
   */
  [[nodiscard]] Legion::FutureMap extract_scalar(const ParallelPolicy& parallel_policy,
                                                 const Legion::FutureMap& result,
                                                 std::size_t offset,
                                                 std::size_t size,
                                                 std::size_t future_size,
                                                 const Legion::Domain& launch_domain) const;
  [[nodiscard]] Legion::Future reduce_future_map(
    const Legion::FutureMap& future_map,
    GlobalRedopID reduction_op,
    const Legion::Future& init_value = Legion::Future{});
  [[nodiscard]] Legion::Future reduce_exception_future_map(const Legion::FutureMap& future_map);

  void issue_release_region_field(
    InternalSharedPtr<LogicalRegionField::PhysicalState> physical_state, bool unordered);
  void issue_discard_field(const Legion::LogicalRegion& region, Legion::FieldID field_id);
  void discard_field(const Legion::LogicalRegion& region, Legion::FieldID field_id);
  void issue_mapping_fence();
  /**
   * @brief Issue a consesus match on discarded fields in multi-rank runs.
   */
  void issue_field_match();
  void issue_execution_fence(bool block = false);
  [[nodiscard]] InternalSharedPtr<LogicalStore> get_timestamp(Timing::Precision precision);
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

  [[nodiscard]] InternalSharedPtr<mapping::detail::Machine> create_toplevel_machine();
  [[nodiscard]] const mapping::detail::Machine& get_machine() const;
  [[nodiscard]] ZStringView get_provenance() const;
  [[nodiscard]] const mapping::detail::LocalMachine& local_machine() const;
  [[nodiscard]] std::uint32_t node_count() const;
  [[nodiscard]] std::uint32_t node_id() const;

  [[nodiscard]] Legion::ProjectionID get_affine_projection(std::uint32_t src_ndim,
                                                           const proj::SymbolicPoint& point);
  [[nodiscard]] Legion::ProjectionID get_delinearizing_projection(
    Span<const std::uint64_t> color_shape);
  [[nodiscard]] Legion::ProjectionID get_compound_projection(Span<const std::uint64_t> color_shape,
                                                             const proj::SymbolicPoint& point);
  [[nodiscard]] Legion::ShardingID get_sharding(const mapping::detail::Machine& machine,
                                                Legion::ProjectionID proj_id);

  [[nodiscard]] Processor get_executing_processor() const;

  [[nodiscard]] Legion::MapperID mapper_id() const;

  [[nodiscard]] bool executing_inline_task() const noexcept;
  void inline_task_start() noexcept;
  void inline_task_end() noexcept;

  [[nodiscard]] static Runtime& get_runtime();
  static void start();
  [[nodiscard]] bool initialized() const;
  void register_shutdown_callback(ShutdownCallback callback);
  [[nodiscard]] std::int32_t finish();
  [[nodiscard]] const Library& core_library() const;

  [[nodiscard]] CUstream get_cuda_stream() const;
  /**
   * @brief Return the current active CUDA device ordinal.
   *
   * This routine may be called from anywhere, including from the top-level task. Inside leaf
   * tasks, it assumes that Realm has set the current context and device for us, and simply
   * returns that.
   *
   * From the top-level task, it attempts to find the "first" GPU assigned to the current
   * processor and returns that.
   *
   * @return The current CUDA device ordinal.
   *
   * @throw std::invalid_argument If Realm fails to detect the current CUDA device.
   * @throw std::runtime_error If this routine is called on a build that does not have CUDA
   * support.
   */
  [[nodiscard]] CUdevice get_current_cuda_device() const;
  [[nodiscard]] cuda::detail::CUDAModuleManager& get_cuda_module_manager();

  [[nodiscard]] Legion::Runtime* get_legion_runtime();
  [[nodiscard]] Legion::Context get_legion_context();

  void start_profiling_range();
  void stop_profiling_range(std::string_view provenance);

 private:
  /**
   * @brief The Legion runtime initialization callback.
   *
   * This routine eventually creates and initializes the main Runtime object. It is registered
   * with Legion to run on every node at startup, and therefore must be called before the
   * runtime is accessed.
   *
   * This routine will also create the core legate library object.
   *
   * @param args The registration args.
   */
  static void initialize_core_library_callback_(const Legion::RegistrationCallbackArgs& args);

  [[nodiscard]] const MapperManager& get_mapper_manager_() const;

  bool initialized_{};
  Legion::Runtime* legion_runtime_{};
  Legion::Context legion_context_{};
  Config config_{};
  std::optional<std::reference_wrapper<Library>> core_library_{};
  std::list<ShutdownCallback> callbacks_{};
  legate::mapping::detail::LocalMachine local_machine_{};

  std::unique_ptr<FieldManager> field_manager_{};
  using RegionManagerKey = Legion::IndexSpace;
  std::unordered_map<RegionManagerKey, RegionManager> region_managers_{};
  std::optional<CommunicatorManager> communicator_manager_{};
  std::optional<PartitionManager> partition_manager_{};
  Scope scope_{};

  std::unordered_map<Domain, Legion::IndexSpace> cached_index_spaces_{};

  using AffineProjectionDesc = std::pair<std::uint32_t, proj::SymbolicPoint>;
  using CompoundProjectionDesc =
    std::pair<SmallVector<std::uint64_t, LEGATE_MAX_DIM>, proj::SymbolicPoint>;
  std::int64_t next_projection_id_{
    static_cast<std::int64_t>(CoreProjectionOp::FIRST_DYNAMIC_FUNCTOR)};
  std::unordered_map<AffineProjectionDesc, Legion::ProjectionID, hasher<AffineProjectionDesc>>
    affine_projections_{};
  std::unordered_map<SmallVector<std::uint64_t, LEGATE_MAX_DIM>,
                     Legion::ProjectionID,
                     hasher<SmallVector<std::uint64_t, LEGATE_MAX_DIM>>>
    delinearizing_projections_{};
  std::unordered_map<CompoundProjectionDesc, Legion::ProjectionID, hasher<CompoundProjectionDesc>>
    compound_projections_{};

  using ShardingDesc = std::pair<Legion::ProjectionID, mapping::ProcessorRange>;
  std::int64_t next_sharding_id_{
    static_cast<std::int64_t>(CoreProjectionOp::FIRST_DYNAMIC_FUNCTOR)};
  std::unordered_map<ShardingDesc, Legion::ShardingID, hasher<ShardingDesc>>
    registered_shardings_{};

  std::deque<InternalSharedPtr<Operation>> operations_{};
  std::atomic<std::uint64_t> cur_op_id_{};

  using RegionFieldID = std::pair<Legion::LogicalRegion, Legion::FieldID>;
  std::uint64_t next_store_id_{1};
  std::uint64_t next_storage_id_{1};
  std::size_t field_reuse_size_{1};

  // This could be a hash map, but kept as an ordered map just in case we may later support
  // library-specific shutdown callbacks that can launch tasks.
  std::map<std::string, Library, std::less<>> libraries_{};

  using ReductionOpTableKey = std::pair<std::uint32_t, std::int32_t>;
  std::unordered_map<ReductionOpTableKey, GlobalRedopID, hasher<ReductionOpTableKey>>
    reduction_ops_{};

  std::vector<std::variant<Legion::Future, ReturnedException>> pending_exceptions_{};
  // Thread-local flag to track whether the current thread is executing an inline task.
  // This must be thread-local because multiple Legion tasks can run in parallel,
  // each potentially launching PhysicalTasks inline on different threads.
  static thread_local bool executing_inline_task_;

  std::optional<MapperManager> mapper_manager_{};

  std::optional<cuda::detail::CUDAModuleManager> cu_mod_manager_{};
};

[[nodiscard]] bool has_started();
[[nodiscard]] bool has_finished();

}  // namespace legate::detail

#include <legate/runtime/detail/runtime.inl>
