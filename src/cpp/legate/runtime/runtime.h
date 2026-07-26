/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/logical_store.h>
#include <legate/data/raw_allocation.h>
#include <legate/data/shape.h>
#include <legate/geometry/rect.h>
#include <legate/mapping/machine.h>
#include <legate/operation/task.h>
#include <legate/runtime/library.h>
#include <legate/runtime/resource.h>
#include <legate/task/variant_options.h>
#include <legate/type/types.h>
#include <legate/utilities/detail/doxygen.h>
#include <legate/utilities/span.h>

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

/**
 * @file
 * @brief Definitions for legate::Runtime and top-level APIs
 */

namespace legate::mapping {

class Mapper;

}  // namespace legate::mapping

namespace legate {

using CUdevice = int;

/**
 * @addtogroup runtime
 * @{
 */

class Scalar;
class Type;

namespace detail {

class Runtime;
class Config;

}  // namespace detail

/**
 * @brief Class that implements the Legate runtime
 *
 * The legate runtime provides common services, including library registration,
 * store creation, operator creation and submission, resource management and scoping,
 * and communicator management. Legate libraries are free of all these details about
 * distributed programming and can focus on their domain logic.
 */
class LEGATE_EXPORT Runtime {
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
   * @param default_options Optional default task variant options
   *
   * @return Library object
   *
   * @throw std::invalid_argument If a library already exists for a given name
   */
  [[nodiscard]] Library create_library(std::string_view library_name,
                                       const ResourceConfig& config            = ResourceConfig{},
                                       std::unique_ptr<mapping::Mapper> mapper = nullptr,
                                       std::map<VariantCode, VariantOptions> default_options = {});
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
   * @param default_options Optional default task variant options
   * @param created Optional pointer to a boolean flag indicating whether the library has been
   * created because of this call
   *
   * @return Context object for the library
   */
  [[nodiscard]] Library find_or_create_library(
    std::string_view library_name,
    const ResourceConfig& config                                 = ResourceConfig{},
    std::unique_ptr<mapping::Mapper> mapper                      = nullptr,
    const std::map<VariantCode, VariantOptions>& default_options = {},
    bool* created                                                = nullptr);

  /**
   * @brief Creates an AutoTask
   *
   * @param library Library to query the task
   * @param task_id Library-local Task ID
   *
   * @return Task object
   */
  [[nodiscard]] AutoTask create_task(Library library, LocalTaskID task_id);
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
                                       LocalTaskID task_id,
                                       const tuple<std::uint64_t>& launch_shape);

  // The following overloads are needed to disambiguate
  //
  // create_task(..., {some, initializer, list});
  //
  // both tuple and Span have implicit constructors from initializer list, so we need to
  // provide an explicit initializer_list overload as well. Ideally we phase out tuple (in
  // which case Span should be the other overload that remains), but until then, we need all 3.
  [[nodiscard]] ManualTask create_task(Library library,
                                       LocalTaskID task_id,
                                       Span<const std::uint64_t> launch_shape);
  [[nodiscard]] ManualTask create_task(Library library,
                                       LocalTaskID task_id,
                                       std::initializer_list<std::uint64_t> launch_shape);

  /**
   * @brief Creates a ManualTask
   *
   * This overload should be used when the lower bounds of the task's launch domain should be
   * non-zero. Note that the upper bounds of the launch domain are inclusive (whereas the
   * `launch_shape` in the other overload is exclusive).
   *
   * @param library Library to query the task
   * @param task_id Library-local Task ID
   * @param launch_domain Launch domain for the task
   *
   * @return Task object
   */
  [[nodiscard]] ManualTask create_task(Library library,
                                       LocalTaskID task_id,
                                       const Rect& launch_domain);
  /**
   * @brief Fills a given store with a constant
   *
   * @param lhs Logical store to fill
   * @param value Logical store that contains the constant value to fill the array with
   */
  void issue_fill(const LogicalStore& lhs, const LogicalStore& value);
  /**
   * @brief Fills a given store with a constant
   *
   * @param lhs Logical store to fill
   * @param value Value to fill the array with
   */
  void issue_fill(const LogicalStore& lhs, const Scalar& value);

  /**
   * @brief Submits an AutoTask for execution
   *
   * Each submitted operation goes through multiple pipeline steps to eventually get scheduled
   * for execution. It's not guaranteed that the submitted operation starts executing immediately.
   *
   * The runtime takes the ownership of the submitted task. Once submitted, the task becomes invalid
   * and is not reusable.
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
   * The runtime takes the ownership of the submitted task. Once submitted, the task becomes invalid
   * and is not reusable.
   *
   * @param task A ManualTask to execute
   */
  void submit(ManualTask&& task);

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
   *
   * @return Logical store
   */
  [[nodiscard]] LogicalStore create_store(const Shape& shape, const Type& type);
  /**
   * @brief Creates a normal store out of a `Scalar` object
   *
   * @param scalar Value of the scalar to create a store with
   * @param shape Shape of the store. The volume must be 1. The call does not block on this shape.
   *
   * @return Logical store
   */
  [[nodiscard]] LogicalStore create_store(const Scalar& scalar, const Shape& shape = Shape{});
  /**
   * @brief Creates a store by attaching to an existing allocation.
   *
   * @see legate::RawAllocation For important instructions regarding the mutability and
   * lifetime management of the attached allocation.
   *
   * @param shape Shape of the store. The call does not block on this shape.
   * @param type Element type.
   * @param buffer Pointer to the beginning of the allocation to attach to; allocation must be
   * contiguous, and cover the entire contents of the store (at least `extents.volume() *
   * type.size()` bytes).
   * @param read_only Whether the allocation is read-only.
   * @param ordering In what order the elements are laid out in the passed buffer.
   *
   * @return Logical store.
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
   * @see legate::RawAllocation For important instructions regarding the mutability and
   * lifetime management of the attached allocation.
   *
   * @param shape Shape of the store. The call does not block on this shape.
   * @param type Element type.
   * @param allocation External allocation descriptor.
   * @param ordering In what order the elements are laid out in the passed allocation.
   *
   * @return Logical store.
   */
  [[nodiscard]] LogicalStore create_store(
    const Shape& shape,
    const Type& type,
    const RawAllocation& allocation,
    const mapping::DimOrdering& ordering = mapping::DimOrdering::c_order());
  /**
   * @brief Creates a store by attaching to multiple existing allocations.
   *
   * External allocations must be read-only.
   *
   * @see legate::RawAllocation For important instructions regarding the mutability and
   * lifetime management of the attached allocation.
   *
   * @param shape Shape of the store. The call can BLOCK on this shape for constructing a store
   * partition.
   * @param tile_shape Shape of tiles.
   * @param type Element type.
   * @param allocations Pairs of external allocation descriptors and sub-store colors.
   * @param ordering In what order the elements are laid out in the passed allocations.
   *
   * @return A pair of a logical store and its partition.
   *
   * @throw std::invalid_argument If any of the external allocations are not read-only.
   */
  [[nodiscard]] std::pair<LogicalStore, LogicalStorePartition> create_store(
    const Shape& shape,
    const tuple<std::uint64_t>& tile_shape,
    const Type& type,
    const std::vector<std::pair<RawAllocation, tuple<std::uint64_t>>>& allocations,
    const mapping::DimOrdering& ordering = mapping::DimOrdering::c_order());

  /**
   * @brief Issues a mapping fence
   *
   * A mapping fence, when issued, blocks mapping of all downstream operations before those
   * preceding the fence get mapped. An `issue_mapping_fence` call returns immediately after the
   * request is submitted to the runtime, and the fence asynchronously goes through the runtime
   * analysis pipeline just like any other Legate operations. The call also flushes the scheduling
   * window for batched execution.
   *
   * Mapping fences only affect how the operations are mapped and do not change their execution
   * order, so they are semantically no-op. Nevertheless, they are sometimes useful when the user
   * wants to control how the resource is consumed by independent tasks. Consider a program with two
   * independent tasks A and B, both of which discard their stores right after their execution.  If
   * the stores are too big to be allocated all at once, mapping A and B in parallel (which can
   * happen because A and B are independent and thus nothing stops them from getting mapped
   * concurrently) can lead to a failure. If a mapping fence exists between the two, the runtime
   * serializes their mapping and can reclaim the memory space from stores that would be discarded
   * after A's execution to create allocations for B.
   */
  void issue_mapping_fence();

  /**
   * @brief Issues an execution fence
   *
   * An execution fence is a join point in the task graph. All operations prior to a fence must
   * finish before any of the subsequent operations start.
   *
   * All execution fences are mapping fences by definition; i.e., an execution fence not only
   * prevents the downstream operations from being mapped ahead of itself but also precedes their
   * execution.
   *
   * @param block When `true`, the control code blocks on the fence and all operations that have
   * been submitted prior to this fence.
   */
  void issue_execution_fence(bool block = false);

  template <typename T>
  void register_shutdown_callback(T&& callback);

  /**
   * @brief Returns the total number of nodes
   *
   * @return Total number of nodes
   */
  [[nodiscard]] std::uint32_t node_count() const;

  /**
   * @brief Returns the current rank
   *
   * @return Rank ID
   */
  [[nodiscard]] std::uint32_t node_id() const;

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

  [[nodiscard]] const detail::Runtime* impl() const;

  // Intentionally not documented, these are only exposed for the python bindings.
  [[nodiscard]] LEGATE_PYTHON_EXPORT void* get_cuda_stream() const;
  [[nodiscard]] LEGATE_PYTHON_EXPORT CUdevice get_current_cuda_device() const;
  LEGATE_PYTHON_EXPORT void synchronize_cuda_stream(void* stream) const;
  [[nodiscard]] LEGATE_PYTHON_EXPORT const detail::Config& config() const;

 private:
  explicit Runtime(detail::Runtime& runtime);
  void register_shutdown_callback_(ShutdownCallback callback);

  detail::Runtime* impl_{};
};

/**
 * @brief Starts the Legate runtime
 *
 * @param argc Argument is ignored.
 * @param argv Argument is ignored.
 *
 * @return Always returns 0
 *
 * @deprecated Use the argument-less version of this function instead: `start()`
 *
 * @see start()
 */
[[deprecated("since 25.01; Use the argument-less version of this function instead")]] LEGATE_EXPORT
  std::int32_t
  start(std::int32_t argc, char* argv[]);

/**
 * @ingroup runtime
 *
 * @brief Starts the Legate runtime
 *
 * This makes the runtime ready to accept requests made via its APIs. It may be called any
 * number of times, only the first call has any effect.
 *
 * @throw ConfigurationError If runtime configuration fails.
 * @throw AutoConfigurationError If the automatic configuration heuristics fail.
 */
LEGATE_EXPORT void start();

/**
 * @brief Checks if the runtime has started.
 *
 * @return `true` if the runtime has started, `false` if the runtime has not started yet or
 * after `finish()` is called.
 */
[[nodiscard]] LEGATE_EXPORT bool has_started();

/**
 * @brief Checks if the runtime has finished.
 *
 * @return `true` if `finish()` has been called, `false` otherwise.
 */
[[nodiscard]] LEGATE_EXPORT bool has_finished();

/**
 * @brief Waits for the runtime to finish
 *
 * The client code must call this to make sure all Legate tasks run
 *
 * @return Non-zero value when the runtime encountered a failure, 0 otherwise
 */
[[nodiscard]] LEGATE_EXPORT std::int32_t finish();

[[deprecated("since 24.11: use legate::finish() instead")]] LEGATE_EXPORT void destroy();

/**
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
LEGATE_EXPORT void register_shutdown_callback(T&& callback);

/**
 * @brief Returns the machine for the current scope
 *
 * @return Machine object
 */
[[nodiscard]] LEGATE_EXPORT mapping::Machine get_machine();

/**
 * @brief Checks if the code is running in a task
 *
 * @return true If the code is running in a task
 * @return false If the code is not running in a task
 */
[[nodiscard]] LEGATE_EXPORT bool is_running_in_task();

/** @} */

}  // namespace legate

#include <legate/runtime/runtime.inl>
