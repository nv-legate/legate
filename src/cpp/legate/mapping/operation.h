/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "legate/data/scalar.h"
#include "legate/mapping/array.h"

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
 * @ingroup mapping
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
   * @brief Returns metadata for the task's scalars
   *
   * @param index Index of the scalar array
   *
   * @return Scalar metadata object
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

  explicit Task(detail::Task* impl);

  Task(const Task&)            = delete;
  Task& operator=(const Task&) = delete;
  Task(Task&&)                 = delete;
  Task& operator=(Task&&)      = delete;

 private:
  [[nodiscard]] detail::Task* impl_() const noexcept;

  detail::Task* pimpl_{};
};

}  // namespace legate::mapping

#include "legate/mapping/operation.inl"
