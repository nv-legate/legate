/* Copyright 2021-2023 NVIDIA Corporation
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
#include "core/operation/task.h"
#include "core/runtime/library.h"
#include "core/runtime/resource.h"
#include "core/task/exception.h"
#include "core/utilities/typedefs.h"

/** @defgroup runtime Runtime and library contexts
 */

namespace legate::mapping {
class Mapper;
}  // namespace legate::mapping

namespace legate {

class Scalar;
class Type;

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

 private:
  static void perform_callback(Legion::RegistrationCallbackFnptr callback);

 public:
  // Configuration settings
  static bool show_progress_requested;
  static bool use_empty_task;
  static bool synchronize_stream_view;
  static bool log_mapping_decisions;
  static bool has_socket_mem;
};

/**
 * @ingroup runtime
 * @brief Class that implements the Legate runtime
 *
 * The legate runtime provides common services, including as library registration,
 * store creation, operator creation and submission, resource management and scoping,
 * and communicator management. Legate libraries are free of all these details about
 * distribute programming and can focus on their domain logics.
 */

namespace detail {
class Runtime;
}  // namespace detail

class Runtime {
 public:
  /**
   * @brief Creates a library
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
   * @return Library object
   *
   * @throw std::invalid_argument If a library already exists for a given name
   */
  Library create_library(const std::string& library_name,
                         const ResourceConfig& config            = ResourceConfig{},
                         std::unique_ptr<mapping::Mapper> mapper = nullptr);
  /**
   * @brief Finds a library
   *
   * @param library_name Library name
   *
   * @return Library object
   *
   * @throw std::out_of_range If no library is found for a given name
   */
  Library find_library(const std::string& library_name) const;
  /**
   * @brief Attempts to find a library.
   *
   * If no library exists for a given name, a null value will be returned
   *
   * @param library_name Library name
   *
   * @return Library object if a library exists for a given name, a null object otherwise
   */
  std::optional<Library> maybe_find_library(const std::string& library_name) const;
  /**
   * @brief Finds or creates a library.
   *
   * The optional configuration and mapper objects are picked up only when the library is created.
   *
   *
   * @param library_name Library name. Must be unique to this library
   * @param config Optional configuration object
   * @param mapper Optional mapper object
   * @param created Optional pointer to a boolean flag indicating whether the library has been
   * created because of this call
   *
   * @return Context object for the library
   */
  Library find_or_create_library(const std::string& library_name,
                                 const ResourceConfig& config            = ResourceConfig{},
                                 std::unique_ptr<mapping::Mapper> mapper = nullptr,
                                 bool* created                           = nullptr);

 public:
  /**
   * @brief Creates an AutoTask
   *
   * @param library Library to query the task
   * @param task_id Library-local Task ID
   *
   * @return Task object
   */
  AutoTask create_task(Library library, int64_t task_id);
  /**
   * @brief Creates a ManualTask
   *
   * @param library Library to query the task
   * @param task_id Library-local Task ID
   * @param launch_shape Launch domain for the task
   *
   * @return Task object
   */
  ManualTask create_task(Library library, int64_t task_id, const Shape& launch_shape);
  /**
   * @brief Issues a copy between stores.
   *
   * The source and target stores must have the same shape.
   *
   * @param target Copy target
   * @param source Copy source
   */
  void issue_copy(LogicalStore target, LogicalStore source);
  /**
   * @brief Issues a gather copy between stores.
   *
   * The indirection store and the target store must have the same shape.
   *
   * @param target Copy target
   * @param source Copy source
   * @param source_indirect Store for source indirection
   */
  void issue_gather(LogicalStore target, LogicalStore source, LogicalStore source_indirect);
  /**
   * @brief Issues a scatter copy between stores.
   *
   * The indirection store and the source store must have the same shape.
   *
   * @param target Copy target
   * @param target_indirect Store for target indirection
   * @param source Copy source
   */
  void issue_scatter(LogicalStore target, LogicalStore target_indirect, LogicalStore source);
  /**
   * @brief Issues a scatter-gather copy between stores.
   *
   * The indirection stores must have the same shape.
   *
   * @param target Copy target
   * @param target_indirect Store for target indirection
   * @param source Copy source
   * @param source_indirect Store for source indirection
   */
  void issue_scatter_gather(LogicalStore target,
                            LogicalStore target_indirect,
                            LogicalStore source,
                            LogicalStore source_indirect);

  /**
   * @brief Fills a given store with a constant
   *
   * @param lhs Logical store to fill
   * @param value Logical store that contains the constant value to fill the store with
   */
  void issue_fill(LogicalStore lhs, LogicalStore value);
  /**
   * @brief Fills a given store with a constant
   *
   * @param lhs Logical store to fill
   * @param value Value to fill the store with
   */
  void issue_fill(LogicalStore lhs, const Scalar& value);
  /**
   * @brief Submits an AutoTask for execution
   *
   * Each submitted operation goes through multiple pipeline steps to eventually get scheduled
   * for execution. It's not guaranteed that the submitted operation starts executing immediately.
   *
   * @param task An AutoTask to execute
   */
  void submit(AutoTask&& task);
  /**
   * @brief Submits a ManualTask for execution
   *
   * Each submitted operation goes through multiple pipeline steps to eventually get scheduled
   * for execution. It's not guaranteed that the submitted operation starts executing immediately.
   *
   * @param task A ManualTask to execute
   */
  void submit(ManualTask&& task);

 public:
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
  /**
   * @brief Returns the machine of the current scope
   *
   * @return Machine object
   */
  mapping::Machine get_machine() const;

 public:
  /**
   * @brief Returns a singleton runtime object
   *
   * @return The runtime object
   */
  static Runtime* get_runtime();
  detail::Runtime* impl() { return impl_; }

 private:
  Runtime(detail::Runtime* runtime);
  ~Runtime();
  detail::Runtime* impl_{nullptr};
};

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
int32_t finish();

/**
 * @brief Returns the machine for the current scope
 *
 * @return Machine object
 */
mapping::Machine get_machine();

}  // namespace legate

#include "core/runtime/runtime.inl"
