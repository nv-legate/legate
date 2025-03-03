/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/scalar.h>
#include <legate/mapping/array.h>
#include <legate/utilities/detail/doxygen.h>

#include <vector>

/**
 * @file
 * @brief Class definitions for operations and stores used in mapping
 */

namespace legate::mapping {

namespace detail {
class Task;
}  // namespace detail

/**
 * @addtogroup mapping
 * @{
 */

/**
 * @brief A metadata class for tasks
 */
class Task {
 public:
  /**
   * @brief Returns the task id
   *
   * @return Task id
   */
  [[nodiscard]] LocalTaskID task_id() const;

  /**
   * @brief Returns metadata for the task's input arrays
   *
   * @return Vector of array metadata objects
   */
  [[nodiscard]] std::vector<Array> inputs() const;
  /**
   * @brief Returns metadata for the task's output arrays
   *
   * @return Vector of array metadata objects
   */
  [[nodiscard]] std::vector<Array> outputs() const;
  /**
   * @brief Returns metadata for the task's reduction arrays
   *
   * @return Vector of array metadata objects
   */
  [[nodiscard]] std::vector<Array> reductions() const;
  /**
   * @brief Returns the vector of the task's by-value arguments. Unlike `mapping::Array`
   * objects that have no access to data in the arrays, the returned `Scalar` objects
   * contain valid arguments to the task
   *
   * @return Vector of `Scalar` objects
   */
  [[nodiscard]] std::vector<Scalar> scalars() const;

  /**
   * @brief Returns metadata for the task's input array
   *
   * @param index Index of the input array
   *
   * @return Array metadata object
   */
  [[nodiscard]] Array input(std::uint32_t index) const;
  /**
   * @brief Returns metadata for the task's output array
   *
   * @param index Index of the output array
   *
   * @return Array metadata object
   */
  [[nodiscard]] Array output(std::uint32_t index) const;
  /**
   * @brief Returns metadata for the task's reduction array
   *
   * @param index Index of the reduction array
   *
   * @return Array metadata object
   */
  [[nodiscard]] Array reduction(std::uint32_t index) const;
  /**
   * @brief Returns a by-value argument of the task
   *
   * @param index Index of the scalar
   *
   * @return Scalar
   */
  [[nodiscard]] Scalar scalar(std::uint32_t index) const;

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
   * @brief Indicates whether the task is parallelized
   *
   * @return true The task is a single task
   * @return false The task is one in a set of multiple parallel tasks
   */
  [[nodiscard]] bool is_single_task() const;
  /**
   * @brief Returns the launch domain
   *
   * @return Launch domain
   */
  [[nodiscard]] const Domain& get_launch_domain() const;

  explicit Task(const detail::Task* impl);

  Task(const Task&)            = delete;
  Task& operator=(const Task&) = delete;
  Task(Task&&)                 = delete;
  Task& operator=(Task&&)      = delete;

  [[nodiscard]] const detail::Task* impl() const noexcept;

 private:
  const detail::Task* pimpl_{};
};

/** @} */

}  // namespace legate::mapping

#include <legate/mapping/operation.inl>
