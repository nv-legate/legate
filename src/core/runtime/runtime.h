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

#include "core/data/logical_array.h"
#include "core/data/logical_store.h"
#include "core/data/shape.h"
#include "core/mapping/machine.h"
#include "core/operation/task.h"
#include "core/runtime/library.h"
#include "core/runtime/resource.h"
#include "core/task/exception.h"
#include "core/type/type_info.h"

#include <memory>
#include <optional>
#include <string>

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
  [[nodiscard]] Library create_library(const std::string& library_name,
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
  [[nodiscard]] Library find_library(const std::string& library_name) const;
  /**
   * @brief Attempts to find a library.
   *
   * If no library exists for a given name, a null value will be returned
   *
   * @param library_name Library name
   *
   * @return Library object if a library exists for a given name, a null object otherwise
   */
  [[nodiscard]] std::optional<Library> maybe_find_library(const std::string& library_name) const;
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
  [[nodiscard]] Library find_or_create_library(const std::string& library_name,
                                               const ResourceConfig& config = ResourceConfig{},
                                               std::unique_ptr<mapping::Mapper> mapper = nullptr,
                                               bool* created                           = nullptr);

  /**
   * @brief Creates an AutoTask
   *
   * @param library Library to query the task
   * @param task_id Library-local Task ID
   *
   * @return Task object
   */
  [[nodiscard]] AutoTask create_task(Library library, int64_t task_id);
  /**
   * @brief Creates a ManualTask
   *
   * @param library Library to query the task
   * @param task_id Library-local Task ID
   * @param launch_shape Launch domain for the task
   *
   * @return Task object
   */
  [[nodiscard]] ManualTask create_task(Library library, int64_t task_id, const Shape& launch_shape);
  /**
   * @brief Issues a copy between stores.
   *
   * The source and target stores must have the same shape.
   *
   * @param target Copy target
   * @param source Copy source
   * @param redop ID of the reduction operator to use (optional). The store's type must support the
   * operator.
   *
   * @throw std::invalid_argument If the store's type doesn't support the reduction operator
   */
  void issue_copy(LogicalStore& target,
                  const LogicalStore& source,
                  std::optional<ReductionOpKind> redop = std::nullopt);
  /**
   * @brief Issues a copy between stores.
   *
   * The source and target stores must have the same shape.
   *
   * @param target Copy target
   * @param source Copy source
   * @param redop ID of the reduction operator to use (optional). The store's type must support the
   * operator.
   *
   * @throw std::invalid_argument If the store's type doesn't support the reduction operator
   */
  void issue_copy(LogicalStore& target, const LogicalStore& source, std::optional<int32_t> redop);
  /**
   * @brief Issues a gather copy between stores.
   *
   * The indirection store and the target store must have the same shape.
   *
   * @param target Copy target
   * @param source Copy source
   * @param source_indirect Store for source indirection
   * @param redop ID of the reduction operator to use (optional). The store's type must support the
   * operator.
   *
   * @throw std::invalid_argument If the store's type doesn't support the reduction operator
   */
  void issue_gather(LogicalStore& target,
                    const LogicalStore& source,
                    const LogicalStore& source_indirect,
                    std::optional<ReductionOpKind> redop = std::nullopt);
  /**
   * @brief Issues a gather copy between stores.
   *
   * The indirection store and the target store must have the same shape.
   *
   * @param target Copy target
   * @param source Copy source
   * @param source_indirect Store for source indirection
   * @param redop ID of the reduction operator to use (optional). The store's type must support the
   * operator.
   *
   * @throw std::invalid_argument If the store's type doesn't support the reduction operator
   */
  void issue_gather(LogicalStore& target,
                    const LogicalStore& source,
                    const LogicalStore& source_indirect,
                    std::optional<int32_t> redop);
  /**
   * @brief Issues a scatter copy between stores.
   *
   * The indirection store and the source store must have the same shape.
   *
   * @param target Copy target
   * @param target_indirect Store for target indirection
   * @param source Copy source
   * @param redop ID of the reduction operator to use (optional). The store's type must support the
   * operator.
   *
   * @throw std::invalid_argument If the store's type doesn't support the reduction operator
   */
  void issue_scatter(LogicalStore& target,
                     LogicalStore& target_indirect,
                     const LogicalStore& source,
                     std::optional<ReductionOpKind> redop = std::nullopt);
  /**
   * @brief Issues a scatter copy between stores.
   *
   * The indirection store and the source store must have the same shape.
   *
   * @param target Copy target
   * @param target_indirect Store for target indirection
   * @param source Copy source
   * @param redop ID of the reduction operator to use (optional). The store's type must support the
   * operator.
   *
   * @throw std::invalid_argument If the store's type doesn't support the reduction operator
   */
  void issue_scatter(LogicalStore& target,
                     LogicalStore& target_indirect,
                     const LogicalStore& source,
                     std::optional<int32_t> redop);
  /**
   * @brief Issues a scatter-gather copy between stores.
   *
   * The indirection stores must have the same shape.
   *
   * @param target Copy target
   * @param target_indirect Store for target indirection
   * @param source Copy source
   * @param source_indirect Store for source indirection
   * @param redop ID of the reduction operator to use (optional). The store's type must support the
   * operator.
   *
   * @throw std::invalid_argument If the store's type doesn't support the reduction operator
   */
  void issue_scatter_gather(LogicalStore& target,
                            LogicalStore& target_indirect,
                            const LogicalStore& source,
                            const LogicalStore& source_indirect,
                            std::optional<ReductionOpKind> redop = std::nullopt);
  /**
   * @brief Issues a scatter-gather copy between stores.
   *
   * The indirection stores must have the same shape.
   *
   * @param target Copy target
   * @param target_indirect Store for target indirection
   * @param source Copy source
   * @param source_indirect Store for source indirection
   * @param redop ID of the reduction operator to use (optional). The store's type must support the
   * operator.
   *
   * @throw std::invalid_argument If the store's type doesn't support the reduction operator
   */
  void issue_scatter_gather(LogicalStore& target,
                            LogicalStore& target_indirect,
                            const LogicalStore& source,
                            const LogicalStore& source_indirect,
                            std::optional<int32_t> redop);
  /**
   * @brief Fills a given array with a constant
   *
   * @param lhs Logical array to fill
   * @param value Logical store that contains the constant value to fill the array with
   */
  void issue_fill(const LogicalArray& lhs, const LogicalStore& value);
  /**
   * @brief Fills a given array with a constant
   *
   * @param lhs Logical array to fill
   * @param value Value to fill the array with
   */
  void issue_fill(const LogicalArray& lhs, const Scalar& value);
  /**
   * @brief tree_reduce given store and task id
   *
   * @param task_id reduction task ID
   * @param store Logical store to reduce
   */
  [[nodiscard]] LogicalStore tree_reduce(Library library,
                                         int64_t task_id,
                                         const LogicalStore& store,
                                         int64_t radix = 4);

  /**
   * @brief Submits an AutoTask for execution
   *
   * Each submitted operation goes through multiple pipeline steps to eventually get scheduled
   * for execution. It's not guaranteed that the submitted operation starts executing immediately.
   *
   * @param task An AutoTask to execute
   */
  void submit(AutoTask task);
  /**
   * @brief Submits a ManualTask for execution
   *
   * Each submitted operation goes through multiple pipeline steps to eventually get scheduled
   * for execution. It's not guaranteed that the submitted operation starts executing immediately.
   *
   * @param task A ManualTask to execute
   */
  void submit(ManualTask task);

  /**
   * @brief Creates an unbound array
   *
   * @param extents Shape of the array
   * @param type Element type
   * @param optimize_scalar When true, the runtime internally uses futures optimized for storing
   * scalars
   *
   * @return Logical array
   */
  [[nodiscard]] LogicalArray create_array(const Type& type,
                                          uint32_t dim  = 1,
                                          bool nullable = false);
  /**
   * @brief Creates a normal array
   *
   * @param extents Shape of the array
   * @param type Element type
   * @param optimize_scalar When true, the runtime internally uses futures optimized for storing
   * scalars
   *
   * @return Logical array
   */
  [[nodiscard]] LogicalArray create_array(const Shape& extents,
                                          const Type& type,
                                          bool nullable        = false,
                                          bool optimize_scalar = false);
  /**
   * @brief Creates an array isomorphic to the given array
   *
   * @param type Optional type for the resulting array. Must be compatible with the input array's
   * type
   *
   * @return Logical array isomorphic to the input
   */
  [[nodiscard]] LogicalArray create_array_like(const LogicalArray& to_mirror,
                                               std::optional<Type> type = std::nullopt);

  /**
   * @brief Creates an unbound store
   *
   * @param type Element type
   * @param dim Number of dimensions of the store
   *
   * @return Logical store
   */
  [[nodiscard]] LogicalStore create_store(const Type& type, uint32_t dim = 1);
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
  [[nodiscard]] LogicalStore create_store(const Shape& extents,
                                          const Type& type,
                                          bool optimize_scalar = false);
  /**
   * @brief Creates a normal store out of a `Scalar` object
   *
   * @param scalar Value of the scalar to create a store with
   * @param extents Shape of the store. The volume must be 1.
   *
   * @return Logical store
   */
  [[nodiscard]] LogicalStore create_store(const Scalar& scalar, const Shape& extents = Shape{1});
  /**
   * @brief Creates a store by attaching to existing memory.
   *
   * If `share` is false, then Legate will make an internal copy of the passed buffer.
   *
   * If `share` is true, then the existing buffer will be reused, and Legate will update it
   * according to any modifications made to the returned store. The caller must keep the buffer
   * alive until they explicitly call `detach` on the result store. The contents of the attached
   * buffer are only guaranteed to be up-to-date after `detach` returns.
   *
   * @param extents Shape of the store
   * @param type Element type
   * @param buffer Pointer to the beginning of the memory to attach to; memory must be contiguous,
   * and cover the entire contents of the store (at least `extents.volume() * type.size()` bytes)
   * @param ordering In what order the elements are laid out in the passed buffer
   * @param share Whether to reuse the passed buffer in-place
   *
   * @return Logical store
   */
  [[nodiscard]] LogicalStore create_store(
    const Shape& extents,
    const Type& type,
    void* buffer,
    bool share                           = false,
    const mapping::DimOrdering& ordering = mapping::DimOrdering::c_order());

  /**
   * @brief Returns the maximum number of pending exceptions
   *
   * @return Maximum number of pending exceptions
   */
  [[nodiscard]] uint32_t max_pending_exceptions() const;
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
  [[nodiscard]] std::optional<TaskException> check_pending_task_exception();

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

  /**
   * @brief Returns the machine of the current scope
   *
   * @return Machine object
   */
  [[nodiscard]] mapping::Machine get_machine() const;

  /**
   * @brief Returns a singleton runtime object
   *
   * @return The runtime object
   */
  [[nodiscard]] static Runtime* get_runtime();

  [[nodiscard]] detail::Runtime* impl();

 private:
  explicit Runtime(detail::Runtime* runtime);

  detail::Runtime* impl_{};
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
[[nodiscard]] int32_t start(int32_t argc, char** argv);

/**
 * @brief Waits for the runtime to finish
 *
 * The client code must call this to make sure all Legate tasks run
 *
 * @return Non-zero value when the runtime encountered a failure, 0 otherwise
 */
[[nodiscard]] int32_t finish();

void destroy();

/**
 * @brief Returns the machine for the current scope
 *
 * @return Machine object
 */
[[nodiscard]] mapping::Machine get_machine();

}  // namespace legate

#include "core/runtime/runtime.inl"
