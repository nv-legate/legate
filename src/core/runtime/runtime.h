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

#include "core/data/external_allocation.h"
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
#include <string_view>
#include <type_traits>

/**
 * @file
 * @brief Definitions for legate::Runtime and top-level APIs
 */

/** @defgroup runtime Runtime and library contexts
 */

namespace legate::mapping {
class Mapper;
}  // namespace legate::mapping

namespace legate {

class Scalar;
class Type;

extern std::uint32_t extract_env(const char* env_name,
                                 std::uint32_t default_value,
                                 std::uint32_t test_value);

namespace detail {
class Runtime;
}  // namespace detail

/**
 * @ingroup runtime
 *
 * @brief Class that implements the Legate runtime
 *
 * The legate runtime provides common services, including as library registration,
 * store creation, operator creation and submission, resource management and scoping,
 * and communicator management. Legate libraries are free of all these details about
 * distribute programming and can focus on their domain logics.
 */
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
  [[nodiscard]] Library create_library(std::string_view library_name,
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
  [[nodiscard]] Library find_library(std::string_view library_name) const;
  /**
   * @brief Attempts to find a library.
   *
   * If no library exists for a given name, a null value will be returned
   *
   * @param library_name Library name
   *
   * @return Library object if a library exists for a given name, a null object otherwise
   */
  [[nodiscard]] std::optional<Library> maybe_find_library(std::string_view library_name) const;
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
  [[nodiscard]] Library find_or_create_library(std::string_view library_name,
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
  [[nodiscard]] AutoTask create_task(Library library, std::int64_t task_id);
  /**
   * @brief Creates a ManualTask
   *
   * @param library Library to query the task
   * @param task_id Library-local Task ID
   * @param launch_shape Launch domain for the task
   *
   * @return Task object
   */
  [[nodiscard]] ManualTask create_task(Library library,
                                       std::int64_t task_id,
                                       const tuple<std::uint64_t>& launch_shape);
  /**
   * @brief Creates a ManualTask
   *
   * This overload should be used when the lower bounds of the task's launch domain should be
   * non-zero. Note that the upper bounds of the launch domain are inclusive (whereas the
   * `launch_shape` in the other overload is exlusive).
   *
   * @param library Library to query the task
   * @param task_id Library-local Task ID
   * @param launch_domain Launch domain for the task
   *
   * @return Task object
   */
  [[nodiscard]] ManualTask create_task(Library library,
                                       std::int64_t task_id,
                                       const Domain& launch_domain);
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
  void issue_copy(LogicalStore& target,
                  const LogicalStore& source,
                  std::optional<std::int32_t> redop);
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
                    std::optional<std::int32_t> redop);
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
                     const LogicalStore& target_indirect,
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
                     const LogicalStore& target_indirect,
                     const LogicalStore& source,
                     std::optional<std::int32_t> redop);
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
                            const LogicalStore& target_indirect,
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
                            const LogicalStore& target_indirect,
                            const LogicalStore& source,
                            const LogicalStore& source_indirect,
                            std::optional<std::int32_t> redop);
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
   * @brief Performs reduction on a given store via a task
   *
   * @param library The library for the reducer task
   * @param task_id reduction task ID
   * @param store Logical store to reduce
   * @param radix Optional radix value that determines the maximum number of input stores to the
   * task at each reduction step
   *
   */
  [[nodiscard]] LogicalStore tree_reduce(Library library,
                                         std::int64_t task_id,
                                         const LogicalStore& store,
                                         std::int32_t radix = 4);

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
   * @param type Element type
   * @param dim Number of dimensions
   * @param nullable Nullability of the array
   *
   * @return Logical array
   */
  [[nodiscard]] LogicalArray create_array(const Type& type,
                                          std::uint32_t dim = 1,
                                          bool nullable     = false);
  /**
   * @brief Creates a normal array
   *
   * @param shape Shape of the array. The call does not block on this shape
   * @param type Element type
   * @param nullable Nullability of the array
   * @param optimize_scalar When true, the runtime internally uses futures optimized for storing
   * scalars
   *
   * @return Logical array
   */
  [[nodiscard]] LogicalArray create_array(const Shape& shape,
                                          const Type& type,
                                          bool nullable        = false,
                                          bool optimize_scalar = false);
  /**
   * @brief Creates an array isomorphic to the given array
   *
   * @param to_mirror The array whose shape would be used to create the output array. The call does
   * not block on the array's shape.
   * @param type Optional type for the resulting array. Must be compatible with the input array's
   * type
   *
   * @return Logical array isomorphic to the input
   */
  [[nodiscard]] LogicalArray create_array_like(const LogicalArray& to_mirror,
                                               std::optional<Type> type = std::nullopt);

  /**
   * @brief Creates a string array from the existing sub-arrays
   *
   * The caller is responsible for making sure that the vardata sub-array is valid for all the
   * descriptors in the descriptor sub-array
   *
   * @param descriptor Sub-array for descriptors
   * @param vardata Sub-array for characters
   *
   * @return String logical array
   *
   * @throw std::invalid_argument When any of the following is true:
   * 1) `descriptor` or `vardata` is unbound or N-D where N > 1
   * 2) `descriptor` does not have a 1D rect type
   * 3) `vardata` is nullable
   * 4) `vardata` does not have an int8 type
   */
  [[nodiscard]] StringLogicalArray create_string_array(const LogicalArray& descriptor,
                                                       const LogicalArray& vardata);

  /**
   * @brief Creates a list array from the existing sub-arrays
   *
   * The caller is responsible for making sure that the vardata sub-array is valid for all the
   * descriptors in the descriptor sub-array
   *
   * @param descriptor Sub-array for descriptors
   * @param vardata Sub-array for vardata
   * @param type Optional list type the returned array would have
   *
   * @return List logical array
   *
   * @throw std::invalid_argument When any of the following is true:
   * 1) `type` is not a list type
   * 2) `descriptor` or `vardata` is unbound or N-D where N > 1
   * 3) `descriptor` does not have a 1D rect type
   * 4) `vardata` is nullable
   * 5) `vardata` and `type` have different element types
   */
  [[nodiscard]] ListLogicalArray create_list_array(const LogicalArray& descriptor,
                                                   const LogicalArray& vardata,
                                                   std::optional<Type> type = std::nullopt);

  /**
   * @brief Creates an unbound store
   *
   * @param type Element type
   * @param dim Number of dimensions of the store
   *
   * @return Logical store
   */
  [[nodiscard]] LogicalStore create_store(const Type& type, std::uint32_t dim = 1);
  /**
   * @brief Creates a normal store
   *
   * @param shape Shape of the store. The call does not block on this shape.
   * @param type Element type
   * @param optimize_scalar When true, the runtime internally uses futures optimized for storing
   * scalars
   *
   * @return Logical store
   */
  [[nodiscard]] LogicalStore create_store(const Shape& shape,
                                          const Type& type,
                                          bool optimize_scalar = false);
  /**
   * @brief Creates a normal store out of a `Scalar` object
   *
   * @param scalar Value of the scalar to create a store with
   * @param shape Shape of the store. The volume must be 1. The call does not block on this shape.
   *
   *
   * @return Logical store
   */
  [[nodiscard]] LogicalStore create_store(const Scalar& scalar, const Shape& shape = Shape{1});
  /**
   * @brief Creates a store by attaching to an existing allocation.
   *
   * This call does not block wait on the input shape
   *
   * @param shape Shape of the store. The call does not block on this shape.
   * @param type Element type
   * @param buffer Pointer to the beginning of the allocation to attach to; allocation must be
   * contiguous, and cover the entire contents of the store (at least `extents.volume() *
   * type.size()` bytes)
   * @param read_only Whether the allocation is read-only
   * @param ordering In what order the elements are laid out in the passed buffer
   *
   * @return Logical store
   */
  [[nodiscard]] LogicalStore create_store(
    const Shape& shape,
    const Type& type,
    void* buffer,
    bool read_only                       = true,
    const mapping::DimOrdering& ordering = mapping::DimOrdering::c_order());
  /**
   * @brief Creates a store by attaching to an existing allocation.
   *
   * @param shape Shape of the store. The call does not block on this shape.
   * @param type Element type
   * @param allocation External allocation descriptor
   * @param ordering In what order the elements are laid out in the passed allocation
   *
   * @return Logical store
   */
  [[nodiscard]] LogicalStore create_store(
    const Shape& shape,
    const Type& type,
    const ExternalAllocation& allocation,
    const mapping::DimOrdering& ordering = mapping::DimOrdering::c_order());
  /**
   * @brief Creates a store by attaching to multiple existing allocations.
   *
   * External allocations must be read-only.
   *
   * @param shape Shape of the store. The call can BLOCK on this shape for constructing a store
   * partition
   * @param tile_shape Shape of tiles
   * @param type Element type
   * @param allocations Pairs of external allocation descriptors and sub-store colors
   * @param ordering In what order the elements are laid out in the passed allocatios
   *
   * @return A pair of a logical store and its partition
   *
   * @throw std::invalid_argument If any of the external allocations are not read-only
   */
  [[nodiscard]] std::pair<LogicalStore, LogicalStorePartition> create_store(
    const Shape& shape,
    const tuple<std::uint64_t>& tile_shape,
    const Type& type,
    const std::vector<std::pair<ExternalAllocation, tuple<std::uint64_t>>>& allocations,
    const mapping::DimOrdering& ordering = mapping::DimOrdering::c_order());

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

  template <typename T>
  void register_shutdown_callback(T&& callback);

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
  void register_shutdown_callback_(ShutdownCallback callback);

  detail::Runtime* impl_{};
};

/**
 * @ingroup runtime
 *
 * @brief Starts the Legate runtime
 *
 * This makes the runtime ready to accept requests made via its APIs
 *
 * @param argc Number of command-line flags
 * @param argv Command-line flags
 *
 * @return Non-zero value when the runtime start-up failed, 0 otherwise
 */
[[nodiscard]] std::int32_t start(std::int32_t argc, char** argv);

/**
 * @ingroup runtime
 *
 * @brief Waits for the runtime to finish
 *
 * The client code must call this to make sure all Legate tasks run
 *
 * @return Non-zero value when the runtime encountered a failure, 0 otherwise
 */
[[nodiscard]] std::int32_t finish();

void destroy();

/**
 * @ingroup runtime
 *
 * @brief Registers a callback that should be invoked during the runtime shutdown
 *
 * Any callbacks will be invoked before the core library and the runtime are destroyed. All
 * callbacks must be non-throwable. Multiple registrations of the same callback are not
 * deduplicated, and thus clients are responsible for registering their callbacks only once if they
 * are meant to be invoked as such. Callbacks are invoked in the FIFO order, and thus any callbacks
 * that are registered by another callback will be added to the end of the list of callbacks.
 * Callbacks can launch tasks and the runtime will make sure of their completion before initializing
 * its shutdown.
 *
 * @param callback A shutdown callback
 */
template <typename T>
void register_shutdown_callback(T&& callback);

/**
 * @ingroup runtime
 *
 * @brief Returns the machine for the current scope
 *
 * @return Machine object
 */
[[nodiscard]] mapping::Machine get_machine();

}  // namespace legate

#include "core/runtime/runtime.inl"
