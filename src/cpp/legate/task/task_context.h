/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/comm/communicator.h>
#include <legate/data/physical_array.h>
#include <legate/data/scalar.h>
#include <legate/mapping/machine.h>
#include <legate/mapping/mapping.h>
#include <legate/utilities/detail/doxygen.h>

#include <string_view>
#include <vector>

struct CUstream_st;

/**
 * @file
 * @brief Class definition for legate::TaskContext
 */

namespace legate::detail {
class TaskContext;
}  // namespace legate::detail

namespace legate {

/**
 * @addtogroup task
 * @{
 */

/**
 * @brief A task context that contains task arguments and communicators
 */
class TaskContext {
 public:
  /**
   * @brief Returns the global ID of the task
   *
   * @return The global task id
   */
  [[nodiscard]] GlobalTaskID task_id() const noexcept;
  /**
   * @brief Returns the Legate variant kind of the task
   *
   * @return The variant kind
   */
  [[nodiscard]] VariantCode variant_kind() const noexcept;
  /**
   * @brief Returns an input array of the task
   *
   * @param index Index of the array
   *
   * @return Array
   */
  [[nodiscard]] PhysicalArray input(std::uint32_t index) const;
  /**
   * @brief Returns all input arrays of the task
   *
   * @return Vector of arrays
   */
  [[nodiscard]] std::vector<PhysicalArray> inputs() const;
  /**
   * @brief Returns an output array of the task
   *
   * @param index Index of the array
   *
   * @return Array
   */
  [[nodiscard]] PhysicalArray output(std::uint32_t index) const;
  /**
   * @brief Returns all output arrays of the task
   *
   * @return Vector of arrays
   */
  [[nodiscard]] std::vector<PhysicalArray> outputs() const;
  /**
   * @brief Returns a reduction array of the task
   *
   * @param index Index of the array
   *
   * @return Array
   */
  [[nodiscard]] PhysicalArray reduction(std::uint32_t index) const;
  /**
   * @brief Returns all reduction arrays of the task
   *
   * @return Vector of arrays
   */
  [[nodiscard]] std::vector<PhysicalArray> reductions() const;
  /**
   * @brief Returns a by-value argument of the task
   *
   * @param index Index of the scalar
   *
   * @return Scalar
   */
  [[nodiscard]] Scalar scalar(std::uint32_t index) const;
  /**
   * @brief Returns by-value arguments of the task
   *
   * @return Vector of scalars
   */
  [[nodiscard]] std::vector<Scalar> scalars() const;
  /**
   * @brief Returns a communicator of the task
   *
   * If a task launch ends up emitting only a single point task, that task will not get passed a
   * communicator, even if one was requested at task launching time. Therefore, tasks using
   * communicators should be prepared to handle the case where the returned vector is empty.
   *
   * @param index Index of the communicator
   *
   * @return Communicator
   */
  [[nodiscard]] const comm::Communicator& communicator(std::uint32_t index) const;
  /**
   * @brief Returns communicators of the task
   *
   * If a task launch ends up emitting only a single point task, that task will not get passed a
   * communicator, even if one was requested at task launching time. Therefore, most tasks using
   * communicators should be prepared to handle the case where the returned vector is empty.
   *
   * @return Vector of communicators
   */
  [[nodiscard]] const std::vector<comm::Communicator>& communicators() const;

  /**
   * @brief Returns the number of task's inputs
   *
   * @return Number of arrays
   */
  [[nodiscard]] std::size_t num_inputs() const;
  /**
   * @brief Returns the number of task's outputs
   *
   * @return Number of arrays
   */
  [[nodiscard]] std::size_t num_outputs() const;
  /**
   * @brief Returns the number of task's reductions
   *
   * @return Number of arrays
   */
  [[nodiscard]] std::size_t num_reductions() const;

  /**
   * @brief Returns the number of `Scalar`s
   *
   * @return Number of `Scalar`s
   */
  [[nodiscard]] std::size_t num_scalars() const;

  /**
   * @brief Returns the number of communicators
   *
   * @return Number of communicators
   */
  [[nodiscard]] std::size_t num_communicators() const;

  /**
   * @brief Indicates whether the task is parallelized
   *
   * @return true The task is a single task
   * @return false The task is one in a set of multiple parallel tasks
   */
  [[nodiscard]] bool is_single_task() const;
  /**
   * @brief Indicates whether the task is allowed to raise an exception
   *
   * @return true The task can raise an exception
   * @return false The task must not raise an exception
   */
  [[nodiscard]] bool can_raise_exception() const;
  /**
   * @brief Returns the point of the task. A 0D point will be returned for a single task.
   *
   * @return The point of the task
   */
  [[nodiscard]] const DomainPoint& get_task_index() const;
  /**
   * @brief Returns the task group's launch domain. A single task returns an empty domain
   *
   * @return The task group's launch domain
   */
  [[nodiscard]] const Domain& get_launch_domain() const;
  /**
   * @brief Returns the kind of processor executing this task
   *
   * @return The processor kind
   */
  [[nodiscard]] mapping::TaskTarget target() const;

  [[nodiscard]] mapping::Machine machine() const;

  [[nodiscard]] std::string_view get_provenance() const;

  /**
   * @brief Perform a blocking barrier across all the leaf tasks in a concurrent task launch.
   *
   * When a leaf task invokes this operation, control will not return to the task until all the
   * leaf tasks in the same launch have executed the same barrier.
   *
   * This is useful e.g. to work around NCCL deadlocks, that can be triggered when another
   * concurrent CUDA operation creates a false dependence or resource conflict with the resident
   * NCCL kernels. By performing a barrier before and after every NCCL collective operation
   * happening inside the leaf tasks in a concurrent task launch, we can effectively isolate the
   * execution of the NCCL collective from all other CUDA work, thus preventing the deadlock. In
   * more detail:
   *
   * - put a barrier before the collective operation
   * - emit the collective operation
   * - ensure that NCCL has actually emitted all its operations on the stream (e.g. `ncclGroupEnd`
   *   has been called, if grouping operations)
   * - perform another barrier
   *
   * @snippet integration/nccl.cu NCCL collective operation
   *
   * This operation can only be performed inside leaf tasks (not on the top-level task), and only in
   * variants that have been declared as concurrent. All leaf tasks in a launch must take part in
   * the barrier (it cannot be done only on a subset of them). Breaking any of the previously stated
   * invariants is a fatal error.
   */
  void concurrent_task_barrier();

  [[nodiscard]] detail::TaskContext* impl() const;

  /**
   * @brief Get the current task CUDA stream.
   *
   * @return The current tasks CUDA stream.
   *
   * All asynchronous stream work performed by a GPU variant must be performed on, or
   * synchronized with the stream returned by this method. Doing asynchronous work on other
   * streams and failing to encode those dependencies (or otherwise synchronizing them) on this
   * stream will result in undefined behavior.
   *
   * If the current task is not a GPU task, or does not have GPU support enabled, this method
   * returns `nullptr`.
   */
  [[nodiscard]] CUstream_st* get_task_stream() const;

  explicit TaskContext(detail::TaskContext* impl);

  TaskContext(const TaskContext&)            = default;
  TaskContext& operator=(const TaskContext&) = default;

  TaskContext(TaskContext&&)            = delete;
  TaskContext& operator=(TaskContext&&) = delete;

 private:
  detail::TaskContext* impl_{};
};

/** @} */

}  // namespace legate

#include <legate/task/task_context.inl>
