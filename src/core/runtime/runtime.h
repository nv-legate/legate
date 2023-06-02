/* Copyright 2021-2022 NVIDIA Corporation
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

#include "legion.h"

#include <memory>

#include "core/data/shape.h"
#include "core/data/store.h"
#include "core/legate_c.h"
#include "core/mapping/machine.h"
#include "core/runtime/resource.h"
#include "core/task/exception.h"
#include "core/type/type_info.h"
#include "core/utilities/typedefs.h"

/** @defgroup runtime Runtime and library contexts
 */

namespace legate {

class LibraryContext;
class Scalar;

namespace mapping {

class Mapper;

}  // namespace mapping

extern uint32_t extract_env(const char* env_name,
                            const uint32_t default_value,
                            const uint32_t test_value);

/**
 * @ingroup runtime
 * @brief A utility class that collects static members shared by all Legate libraries
 */
struct Core {
 public:
  static void parse_config(void);
  static void shutdown(void);
  static void show_progress(const Legion::Task* task,
                            Legion::Context ctx,
                            Legion::Runtime* runtime);
  static void report_unexpected_exception(const Legion::Task* task, const TaskException& e);
  static void retrieve_tunable(Legion::Context legion_context,
                               Legion::Runtime* legion_runtime,
                               LibraryContext* context);

 public:
  /**
   * @brief Type signature for registration callbacks
   */
  using RegistrationCallback = void (*)();

  /**
   * @brief Performs a registration callback. Libraries must perform
   * registration of tasks and other components through this function.
   *
   * @tparam CALLBACK Registration callback to perform
   */
  template <RegistrationCallback CALLBACK>
  static void perform_registration();

 public:
  // Configuration settings
  static bool show_progress_requested;
  static bool use_empty_task;
  static bool synchronize_stream_view;
  static bool log_mapping_decisions;
  static bool has_socket_mem;
};

class AutoTask;
class CommunicatorManager;
class FieldManager;
class LogicalRegionField;
class LogicalStore;
class MachineManager;
class ManualTask;
class Operation;
class PartitionManager;
class ProvenanceManager;
class RegionManager;
class Tiling;

/**
 * @ingroup runtime
 * @brief Class that implements the Legate runtime
 *
 * The legate runtime provides common services, including as library registration,
 * store creation, operator creation and submission, resource management and scoping,
 * and communicator management. Legate libraries are free of all these details about
 * distribute programming and can focus on their domain logics.
 */
class Runtime {
 public:
  Runtime(Legion::Runtime* legion_runtime);
  ~Runtime();

 public:
  friend void initialize(int32_t argc, char** argv);
  friend int32_t start(int32_t argc, char** argv);

 public:
  /**
   * @brief Find a library
   *
   * @param library_name Library name
   * @param can_fail Optional flag indicating that the query can fail. When it's true and no
   * library is found for a given name, `nullptr` is returned.
   *
   * @return Context object for the library
   *
   * @throw std::out_of_range If no library is found for a given name and `can_fail` is `false`
   */
  LibraryContext* find_library(const std::string& library_name, bool can_fail = false) const;
  /**
   * @brief Create a library
   *
   * A library is a collection of tasks and custom reduction operators. The maximum number of
   * tasks and reduction operators can be optionally specified with a `ResourceConfig` object.
   * Each library can optionally have a mapper that specifies mapping policies for its tasks.
   * When no mapper is given, the default mapper is used.
   *
   * @param library_name Library name. Must be unique to this library
   * @param config Optional configuration object
   * @param mapper Optional mapper object
   *
   * @throw std::invalid_argument If a library already exists for a given name
   *
   * @return Context object for the library
   */
  LibraryContext* create_library(const std::string& library_name,
                                 const ResourceConfig& config            = ResourceConfig{},
                                 std::unique_ptr<mapping::Mapper> mapper = nullptr);

 public:
  uint32_t get_type_uid();
  void record_reduction_operator(int32_t type_uid, int32_t op_kind, int32_t legion_op_id);
  int32_t find_reduction_operator(int32_t type_uid, int32_t op_kind) const;

 public:
  void enter_callback();
  void exit_callback();
  bool is_in_callback() const;

 public:
  void post_startup_initialization(Legion::Context legion_context);

 public:
  template <typename T>
  T get_tunable(Legion::MapperID mapper_id, int64_t tunable_id);

 public:
  mapping::MachineDesc slice_machine_for_task(LibraryContext* library, int64_t task_id);
  /**
   * @brief Create an AutoTask
   *
   * @param library Library to query the task
   * @param task_id Library-local Task ID
   *
   * @return Task object
   */
  std::unique_ptr<AutoTask> create_task(LibraryContext* library, int64_t task_id);
  /**
   * @brief Create a ManualTask
   *
   * @param library Library to query the task
   * @param task_id Library-local Task ID
   * @param launch_shape Launch domain for the task
   *
   * @return Task object
   */
  std::unique_ptr<ManualTask> create_task(LibraryContext* library,
                                          int64_t task_id,
                                          const Shape& launch_shape);
  void flush_scheduling_window();
  /**
   * @brief Submits an operation for execution
   *
   * Each submitted operation goes through multiple pipeline steps to eventually get scheduled
   * for execution. It's not guaranteed that the submitted operation starts executing immediately.
   *
   * @param op Operation to execute
   */
  void submit(std::unique_ptr<Operation> op);

 public:
  /**
   * @brief Creates an unbound store
   *
   * @param type Element type
   * @param dim Number of dimensions of the store
   *
   * @return Logical store
   */
  LogicalStore create_store(std::unique_ptr<Type> type, int32_t dim = 1);
  /**
   * @brief Creates an unbound store
   *
   * @param type Element type
   * @param dim Number of dimensions of the store
   *
   * @return Logical store
   */
  LogicalStore create_store(const Type& type, int32_t dim = 1);
  /**
   * @brief Creates a normal store
   *
   * @param extents Shape of the store
   * @param type Element type
   * @param optimize_scalar When true, the runtime internally uses futures optimized for storing
   * scalars
   *
   * @return Logical store
   */
  LogicalStore create_store(const Shape& extents,
                            std::unique_ptr<Type> type,
                            bool optimize_scalar = false);
  /**
   * @brief Creates a normal store
   *
   * @param extents Shape of the store
   * @param type Element type
   * @param optimize_scalar When true, the runtime internally uses futures optimized for storing
   * scalars
   *
   * @return Logical store
   */
  LogicalStore create_store(const Shape& extents, const Type& type, bool optimize_scalar = false);
  /**
   * @brief Creates a normal store out of a `Scalar` object
   *
   * @param scalar Value of the scalar to create a store with
   *
   * @return Logical store
   */
  LogicalStore create_store(const Scalar& scalar);

 public:
  /**
   * @brief Returns the maximum number of pending exceptions
   *
   * @return Maximum number of pending exceptions
   */
  uint32_t max_pending_exceptions() const;
  /**
   * @brief Updates the maximum number of pending exceptions
   *
   * If the new maximum number of pending exceptions is smaller than the previous value,
   * `raise_pending_task_exception` will be invoked.
   *
   * @param max_pending_exceptions A new maximum number of pending exceptions
   */
  void set_max_pending_exceptions(uint32_t max_pending_exceptions);
  /**
   * @brief Inspects all pending exceptions and immediately raises the first one if there exists any
   */
  void raise_pending_task_exception();
  /**
   * @brief Returns the first pending exception.
   */
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
  RegionField map_region_field(LibraryContext* context, const LogicalRegionField& region_field);
  void unmap_physical_region(Legion::PhysicalRegion pr);

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

 public:
  Legion::Future dispatch(Legion::TaskLauncher* launcher,
                          std::vector<Legion::OutputRequirement>* output_requirements = nullptr);
  Legion::FutureMap dispatch(Legion::IndexTaskLauncher* launcher,
                             std::vector<Legion::OutputRequirement>* output_requirements = nullptr);

 public:
  Legion::Future extract_scalar(const Legion::Future& result, uint32_t idx) const;
  Legion::FutureMap extract_scalar(const Legion::FutureMap& result,
                                   uint32_t idx,
                                   const Legion::Domain& launch_domain) const;
  Legion::Future reduce_future_map(const Legion::FutureMap& future_map, int32_t reduction_op) const;
  Legion::Future reduce_exception_future_map(const Legion::FutureMap& future_map) const;

 public:
  /**
   * @brief Issues an execution fence
   *
   * An execution fence is a join point in the task graph. All operations prior to a fence must
   * finish before any of the subsequent operations start.
   *
   * @param block When `true`, the control code blocks on the fence and all operations that have
   * been submitted prior to this fence.
   */
  void issue_execution_fence(bool block = false);

 public:
  void initialize_toplevel_machine();
  /**
   * @brief Returns the machine of the current scope
   *
   * @return Machine object
   */
  const mapping::MachineDesc& get_machine() const;

 public:
  Legion::ProjectionID get_projection(int32_t src_ndim, const proj::SymbolicPoint& point);
  Legion::ProjectionID get_delinearizing_projection();

 private:
  void schedule(std::vector<std::unique_ptr<Operation>> operations);

 public:
  static void initialize(int32_t argc, char** argv);
  static int32_t start(int32_t argc, char** argv);

 public:
  /**
   * @brief Returns a singleton runtime object
   *
   * @return The runtime object
   */
  static Runtime* get_runtime();
  static void create_runtime(Legion::Runtime* legion_runtime);
  int32_t wait_for_shutdown();

 private:
  static Runtime* runtime_;

 private:
  Legion::Runtime* legion_runtime_;
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
  std::map<ProjectionDesc, Legion::ProjectionID> registered_projections_;

 private:
  std::vector<std::unique_ptr<Operation>> operations_;
  size_t window_size_{1};
  uint64_t next_unique_id_{0};

 private:
  using RegionFieldID = std::pair<Legion::LogicalRegion, Legion::FieldID>;
  std::map<RegionFieldID, Legion::PhysicalRegion> inline_mapped_;
  uint64_t next_store_id_{1};
  uint64_t next_storage_id_{1};

 private:
  bool in_callback_{false};

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

/**
 * @brief Initializes the Legate runtime
 *
 * @param argc Number of command-line flags
 * @param argv Command-line flags
 */
void initialize(int32_t argc, char** argv);

/**
 * @brief Starts the Legate runtime
 *
 * This makes the runtime ready to accept requests made via its APIs
 *
 * @param argc Number of command-line flags
 * @param argv Command-line flags
 *
 * @return Non-zero value when the runtime start-up failed, 0 otherwise
 */
int32_t start(int32_t argc, char** argv);

/**
 * @brief Waits for the runtime to finish
 *
 * The client code must call this to make sure all Legate tasks run
 *
 * @return Non-zero value when the runtime encountered a failure, 0 otherwise
 */
int32_t wait_for_shutdown();

}  // namespace legate

#include "core/runtime/runtime.inl"
