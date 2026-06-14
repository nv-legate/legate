/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/comm/coll_comm.h>
#include <legate/cuda/detail/cuda_driver_types.h>
#include <legate/cuda/detail/module_manager.h>
#include <legate/data/detail/scalar.h>
#include <legate/mapping/detail/mapping.h>
#include <legate/mapping/machine.h>
#include <legate/operation/detail/timing.h>
#include <legate/runtime/detail/communicator_manager.h>
#include <legate/runtime/detail/config.h>
#include <legate/runtime/detail/library.h>
#include <legate/runtime/detail/partition_manager.h>
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
#include <nccl.h>
#include <optional>
#include <stack>
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
class ManualTask;
class LogicalStore;
class LogicalStorePartition;
class Operation;
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
  void issue_fill(InternalSharedPtr<LogicalStore> lhs, InternalSharedPtr<LogicalStore> value);
  void issue_fill(InternalSharedPtr<LogicalStore> lhs, Scalar value);

  // also used for offloading stores
  void offload_to(mapping::StoreTarget target_mem, const InternalSharedPtr<LogicalStore>& store);

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

 public:
  [[nodiscard]] InternalSharedPtr<LogicalStore> create_store(InternalSharedPtr<Type> type,
                                                             std::uint32_t dim);
  [[nodiscard]] InternalSharedPtr<LogicalStore> create_store(InternalSharedPtr<Shape> shape,
                                                             InternalSharedPtr<Type> type);
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

  [[nodiscard]] CommunicatorManager& communicator_manager();
  [[nodiscard]] const CommunicatorManager& communicator_manager() const;
  [[nodiscard]] PartitionManager& partition_manager();
  [[nodiscard]] const PartitionManager& partition_manager() const;
  [[nodiscard]] Scope& scope();
  [[nodiscard]] const Scope& scope() const;

  void issue_mapping_fence();
  void issue_execution_fence(bool block = false);
  [[nodiscard]] InternalSharedPtr<LogicalStore> get_timestamp(Timing::Precision precision);

  [[nodiscard]] InternalSharedPtr<mapping::detail::Machine> create_toplevel_machine();
  [[nodiscard]] const mapping::detail::Machine& get_machine() const;
  [[nodiscard]] ZStringView get_provenance() const;
  [[nodiscard]] const mapping::detail::LocalMachine& local_machine() const;
  [[nodiscard]] std::uint32_t node_count() const;
  [[nodiscard]] std::uint32_t node_id() const;
  [[nodiscard]] bool single_process() const;

  [[nodiscard]] Processor get_executing_processor() const;

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
  [[nodiscard]] legate::comm::coll::CollComm get_cpu_comm() const;
  [[nodiscard]] ncclComm_t get_gpu_comm() const;

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

  bool initialized_{};
  Config config_{};
  std::optional<std::reference_wrapper<Library>> core_library_{};
  std::list<ShutdownCallback> callbacks_{};
  legate::mapping::detail::LocalMachine local_machine_{};
  CUstream current_stream_{};
  CUdevice current_device_{};
  std::unique_ptr<legate::comm::coll::Coll_Comm> cpu_communicator_{};
  ncclComm_t gpu_communicator_{};

  std::optional<CommunicatorManager> communicator_manager_{};
  std::optional<PartitionManager> partition_manager_{};
  Scope scope_;

  std::mutex operations_lock_{};
  std::list<InternalSharedPtr<Operation>> operations_{};
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

  std::optional<cuda::detail::CUDAModuleManager> cu_mod_manager_{};
};

[[nodiscard]] bool has_started();
[[nodiscard]] bool has_finished();

}  // namespace legate::detail

#include <legate/runtime/detail/runtime.inl>
