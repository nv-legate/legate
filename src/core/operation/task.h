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
#include "core/data/scalar.h"
#include "core/operation/projection.h"
#include "core/partitioning/constraint.h"
#include "core/utilities/internal_shared_ptr.h"
#include "core/utilities/shared_ptr.h"

#include <string>
#include <type_traits>

/**
 * @file
 * @brief Class definitions for legate::AutoTask and legate::ManualTask
 */

namespace legate::detail {
class AutoTask;
class ManualTask;
}  // namespace legate::detail

namespace legate {

/**
 * @ingroup op
 * @brief A class for auto-parallelized task desciptors
 */
class AutoTask {
 public:
  /**
   * @brief Adds an array to the task as input
   *
   * Partitioning of the array is controlled by constraints on the partition symbol
   * associated with the array
   *
   * @param array An array to add to the task as input
   *
   * @return The partition symbol assigned to the array
   */
  Variable add_input(const LogicalArray& array);
  /**
   * @brief Adds an array to the task as output
   *
   * Partitioning of the array is controlled by constraints on the partition symbol
   * associated with the array
   *
   * @param array An array to add to the task as output
   *
   * @return The partition symbol assigned to the array
   */
  Variable add_output(const LogicalArray& array);
  /**
   * @brief Adds an array to the task for reductions
   *
   * Partitioning of the array is controlled by constraints on the partition symbol
   * associated with the array
   *
   * @param array An array to add to the task for reductions
   * @param redop ID of the reduction operator to use. The array's type must support the operator.
   *
   * @return The partition symbol assigned to the array
   */
  Variable add_reduction(const LogicalArray& array, ReductionOpKind redop);
  /**
   * @brief Adds an array to the task for reductions
   *
   * Partitioning of the array is controlled by constraints on the partition symbol
   * associated with the array
   *
   * @param array An array to add to the task for reductions
   * @param redop ID of the reduction operator to use. The array's type must support the operator.
   *
   * @return The partition symbol assigned to the array
   */
  Variable add_reduction(const LogicalArray& array, int32_t redop);

  /**
   * @brief Adds an array to the task as input
   *
   * Partitioning of the array is controlled by constraints on the partition symbol
   * associated with the array
   *
   * @param array An array to add to the task as input
   * @param partition_symbol A partition symbol for the array
   *
   * @return The partition symbol assigned to the array
   */
  Variable add_input(const LogicalArray& array, Variable partition_symbol);
  /**
   * @brief Adds an array to the task as output
   *
   * Partitioning of the array is controlled by constraints on the partition symbol
   * associated with the array
   *
   * @param array An array to add to the task as output
   * @param partition_symbol A partition symbol for the array
   *
   * @return The partition symbol assigned to the array
   */
  Variable add_output(const LogicalArray& array, Variable partition_symbol);
  /**
   * @brief Adds an array to the task for reductions
   *
   * Partitioning of the array is controlled by constraints on the partition symbol
   * associated with the array
   *
   * @param array An array to add to the task for reductions
   * @param redop ID of the reduction operator to use. The array's type must support the operator.
   * @param partition_symbol A partition symbol for the array
   *
   * @return The partition symbol assigned to the array
   */
  Variable add_reduction(const LogicalArray& array,
                         ReductionOpKind redop,
                         Variable partition_symbol);
  /**
   * @brief Adds an array to the task for reductions
   *
   * Partitioning of the array is controlled by constraints on the partition symbol
   * associated with the array
   *
   * @param array An array to add to the task for reductions
   * @param redop ID of the reduction operator to use. The array's type must support the operator.
   * @param partition_symbol A partition symbol for the array
   *
   * @return The partition symbol assigned to the array
   */
  Variable add_reduction(const LogicalArray& array, int32_t redop, Variable partition_symbol);
  /**
   * @brief Adds a by-value scalar argument to the task
   *
   * @param scalar A scalar to add to the task
   */
  void add_scalar_arg(const Scalar& scalar);
  /**
   * @brief Adds a by-value scalar argument to the task
   *
   * @param scalar A scalar to add to the task
   */
  void add_scalar_arg(Scalar&& scalar);

  /**
   @brief Template wrapper for arithmetic types to make adding scalar args
          less verbose

   @tparam T The arithmetic type
   @param t The arithmetic scalar to convert to Scalar
   */
  template <class T, typename = std::enable_if_t<std::is_constructible<Scalar, T>::value>>
  void add_scalar_arg(T&& t);

  /**
   * @brief Adds a partitioning constraint to the task
   *
   * @param constraint A partitioning constraint
   */
  void add_constraint(const Constraint& constraint);

  /**
   * @brief Finds or creates a partition symbol for the given array
   *
   * @param array Array for which the partition symbol is queried
   *
   * @return The existing symbol if there is one for the array, a fresh symbol otherwise
   */
  [[nodiscard]] Variable find_or_declare_partition(const LogicalArray& array);
  /**
   * @brief Declares partition symbol
   *
   * @return A new symbol that can be used when passing an array to an operation
   */
  [[nodiscard]] Variable declare_partition();
  /**
   * @brief Returns the provenance information of this operation
   *
   * @return Provenance
   */
  [[nodiscard]] const std::string& provenance() const;

  /**
   * @brief Sets whether the task needs a concurrent task launch.
   *
   * Any task with at least one communicator will implicitly use concurrent task launch, so this
   * method is to be used when the task needs a concurrent task launch for a reason unknown to
   * Legate.
   *
   * @param concurrent A boolean value indicating whether the task needs a concurrent task launch
   */
  void set_concurrent(bool concurrent);
  /**
   * @brief Sets whether the task has side effects or not.
   *
   * A task is assumed to be free of side effects by default if the task only has scalar arguments.
   *
   * @param has_side_effect A boolean value indicating whether the task has side effects
   */
  void set_side_effect(bool has_side_effect);
  /**
   * @brief Sets whether the task can throw an exception or not.
   *
   * @param can_throw_exception A boolean value indicating whether the task can throw an exception
   */
  void throws_exception(bool can_throw_exception);
  /**
   * @brief Requests a communicator for this task.
   *
   * @param name The name of the communicator to use for this task
   */
  void add_communicator(const std::string& name);

  AutoTask()                               = default;
  AutoTask(AutoTask&&) noexcept            = default;
  AutoTask& operator=(AutoTask&&) noexcept = default;
  AutoTask(const AutoTask&)                = default;
  AutoTask& operator=(const AutoTask&)     = default;
  ~AutoTask() noexcept;

 private:
  friend class Runtime;
  explicit AutoTask(InternalSharedPtr<detail::AutoTask> impl);
  SharedPtr<detail::AutoTask> impl_{};
};

/**
 * @ingroup op
 * @brief A class for manually parallelized task descriptors
 */
class ManualTask {
 public:
  /**
   * @brief Adds a store to the task as input
   *
   * The store will be unpartitioned but broadcasted to all the tasks
   *
   * @param store A store to add to the task as input
   */
  void add_input(const LogicalStore& store);
  /**
   * @brief Adds a store partition to the task as input
   *
   * @param store_partition A store partition to add to the task as input
   * @param projection An optional symbolic point describing a mapping between points in the
   * launch domain and substores in the partition
   */
  void add_input(const LogicalStorePartition& store_partition,
                 std::optional<SymbolicPoint> projection = std::nullopt);
  /**
   * @brief Adds a store to the task as output
   *
   * The store will be unpartitioned but broadcasted to all the tasks
   *
   * @param store A store to add to the task as output
   */
  void add_output(const LogicalStore& store);
  /**
   * @brief Adds a store partition to the task as output
   *
   * @param store_partition A store partition to add to the task as output
   * @param projection An optional symbolic point describing a mapping between points in the
   * launch domain and substores in the partition
   */
  void add_output(const LogicalStorePartition& store_partition,
                  std::optional<SymbolicPoint> projection = std::nullopt);
  /**
   * @brief Adds a store to the task for reductions
   *
   * The store will be unpartitioned but broadcasted to all the tasks
   *
   * @param store A store to add to the task for reductions
   * @param redop ID of the reduction operator to use. The store's type must support the operator.
   */
  void add_reduction(const LogicalStore& store, ReductionOpKind redop);
  /**
   * @brief Adds a store to the task for reductions
   *
   * The store will be unpartitioned but broadcasted to all the tasks
   *
   * @param store A store to add to the task for reductions
   * @param redop ID of the reduction operator to use. The store's type must support the operator.
   */
  void add_reduction(const LogicalStore& store, int32_t redop);
  /**
   * @brief Adds a store partition to the task for reductions
   *
   * @param store_partition A store partition to add to the task for reductions
   * @param redop ID of the reduction operator to use. The store's type must support the operator.
   * @param projection An optional symbolic point describing a mapping between points in the
   * launch domain and substores in the partition
   */
  void add_reduction(const LogicalStorePartition& store_partition,
                     ReductionOpKind redop,
                     std::optional<SymbolicPoint> projection = std::nullopt);
  /**
   * @brief Adds a store partition to the task for reductions
   *
   * @param store_partition A store partition to add to the task for reductions
   * @param redop ID of the reduction operator to use. The store's type must support the operator.
   * @param projection An optional symbolic point describing a mapping between points in the
   * launch domain and substores in the partition
   */
  void add_reduction(const LogicalStorePartition& store_partition,
                     int32_t redop,
                     std::optional<SymbolicPoint> projection = std::nullopt);
  /**
   * @brief Adds a by-value scalar argument to the task
   *
   * @param scalar A scalar to add to the task
   */
  void add_scalar_arg(const Scalar& scalar);
  /**
   * @brief Adds a by-value scalar argument to the task
   *
   * @param scalar A scalar to add to the task
   */
  void add_scalar_arg(Scalar&& scalar);

  /**
   @brief Template wrapper for arithmetic types to make adding scalar args
          less verbose

   @tparam T The arithmetic type
   @param t The arithmetic scalar to convert to Scalar
   */
  template <class T, typename = std::enable_if_t<std::is_constructible<Scalar, T>::value>>
  void add_scalar_arg(T&& t);

  /**
   * @brief Returns the provenance information of this operation
   *
   * @return Provenance
   */
  [[nodiscard]] const std::string& provenance() const;

  /**
   * @brief Sets whether the task needs a concurrent task launch.
   *
   * Any task with at least one communicator will implicitly use concurrent task launch, so this
   * method is to be used when the task needs a concurrent task launch for a reason unknown to
   * Legate.
   *
   * @param concurrent A boolean value indicating whether the task needs a concurrent task launch
   */
  void set_concurrent(bool concurrent);
  /**
   * @brief Sets whether the task has side effects or not.
   *
   * A task is assumed to be free of side effects by default if the task only has scalar arguments.
   *
   * @param has_side_effect A boolean value indicating whether the task has side effects
   */
  void set_side_effect(bool has_side_effect);
  /**
   * @brief Sets whether the task can throw an exception or not.
   *
   * @param can_throw_exception A boolean value indicating whether the task can throw an exception
   */
  void throws_exception(bool can_throw_exception);
  /**
   * @brief Requests a communicator for this task.
   *
   * @param name The name of the communicator to use for this task
   */
  void add_communicator(const std::string& name);

  ~ManualTask() noexcept;
  ManualTask()                                 = default;
  ManualTask(ManualTask&&) noexcept            = default;
  ManualTask& operator=(ManualTask&&) noexcept = default;
  ManualTask(const ManualTask&)                = default;
  ManualTask& operator=(const ManualTask&)     = default;

 private:
  friend class Runtime;
  explicit ManualTask(InternalSharedPtr<detail::ManualTask> impl);
  SharedPtr<detail::ManualTask> impl_{};
};

}  // namespace legate

#include "core/operation/task.inl"
