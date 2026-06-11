/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/scalar.h>
#include <legate/mapping/store.h>
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
class LEGATE_EXPORT Task {
 public:
  /**
   * @brief Returns the task id
   *
   * @return Task id
   */
  [[nodiscard]] LocalTaskID task_id() const;

  /**
   * @brief Returns metadata for the task's input stores
   *
   * @return Vector of store metadata objects
   */
  [[nodiscard]] std::vector<Store> inputs() const;
  /**
   * @brief Returns metadata for the task's output stores
   *
   * @return Vector of store metadata objects
   */
  [[nodiscard]] std::vector<Store> outputs() const;
  /**
   * @brief Returns metadata for the task's reduction stores
   *
   * @return Vector of store metadata objects
   */
  [[nodiscard]] std::vector<Store> reductions() const;
  /**
   * @brief Returns the vector of the task's by-value arguments. Unlike `mapping::Store`
   * objects that have no access to data in the stores, the returned `Scalar` objects
   * contain valid arguments to the task
   *
   * @return Vector of `Scalar` objects
   */
  [[nodiscard]] std::vector<Scalar> scalars() const;

  /**
   * @brief Returns metadata for the task's input store
   *
   * @param index Index of the input store
   *
   * @return Store metadata object
   */
  [[nodiscard]] Store input(std::uint32_t index) const;
  /**
   * @brief Returns metadata for the task's output store
   *
   * @param index Index of the output store
   *
   * @return Store metadata object
   */
  [[nodiscard]] Store output(std::uint32_t index) const;
  /**
   * @brief Returns metadata for the task's reduction store
   *
   * @param index Index of the reduction store
   *
   * @return Store metadata object
   */
  [[nodiscard]] Store reduction(std::uint32_t index) const;
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
   * @return Number of input stores
   */
  [[nodiscard]] std::size_t num_inputs() const;
  /**
   * @brief Returns the number of task's outputs
   *
   * @return Number of output stores
   */
  [[nodiscard]] std::size_t num_outputs() const;
  /**
   * @brief Returns the number of task's reductions
   *
   * @return Number of reduction stores
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
