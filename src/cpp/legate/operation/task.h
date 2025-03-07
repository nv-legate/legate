/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/data/logical_array.h>
#include <legate/data/logical_store.h>
#include <legate/data/scalar.h>
#include <legate/operation/projection.h>
#include <legate/partitioning/constraint.h>
#include <legate/utilities/detail/doxygen.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/shared_ptr.h>

#include <string_view>
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
 * @addtogroup task
 * @{
 */

/**
 * @brief A class for auto-parallelized task descriptors
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
  Variable add_input(LogicalArray array);
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
  Variable add_output(LogicalArray array);
  /**
   * @brief Adds an array to the task for reductions
   *
   * Partitioning of the array is controlled by constraints on the partition symbol
   * associated with the array
   *
   * @param array An array to add to the task for reductions
   * @param redop_kind ID of the reduction operator to use. The array's type must support the
   * operator.
   *
   * @return The partition symbol assigned to the array
   */
  Variable add_reduction(LogicalArray array, ReductionOpKind redop_kind);
  /**
   * @brief Adds an array to the task for reductions
   *
   * Partitioning of the array is controlled by constraints on the partition symbol
   * associated with the array
   *
   * @param array An array to add to the task for reductions
   * @param redop_kind ID of the reduction operator to use. The array's type must support the
   * operator.
   *
   * @return The partition symbol assigned to the array
   */
  Variable add_reduction(LogicalArray array, std::int32_t redop_kind);

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
  Variable add_input(LogicalArray array, Variable partition_symbol);
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
  Variable add_output(LogicalArray array, Variable partition_symbol);
  /**
   * @brief Adds an array to the task for reductions
   *
   * Partitioning of the array is controlled by constraints on the partition symbol
   * associated with the array
   *
   * @param array An array to add to the task for reductions
   * @param redop_kind ID of the reduction operator to use. The array's type must support the
   * operator.
   * @param partition_symbol A partition symbol for the array
   *
   * @return The partition symbol assigned to the array
   */
  Variable add_reduction(LogicalArray array, ReductionOpKind redop_kind, Variable partition_symbol);
  /**
   * @brief Adds an array to the task for reductions
   *
   * Partitioning of the array is controlled by constraints on the partition symbol
   * associated with the array
   *
   * @param array An array to add to the task for reductions
   * @param redop_kind ID of the reduction operator to use. The array's type must support the
   * operator.
   * @param partition_symbol A partition symbol for the array
   *
   * @return The partition symbol assigned to the array
   */
  Variable add_reduction(LogicalArray array, std::int32_t redop_kind, Variable partition_symbol);
  /**
   * @brief Adds a by-value scalar argument to the task
   *
   * @param scalar The Scalar to add to the task
   */
  void add_scalar_arg(const Scalar& scalar);
  /**
   * @brief Adds a by-value scalar argument to the task
   *
   * @tparam T The scalar value's type. Scalar must be constructible from a value of T
   * @param value The scalar value to convert to Scalar and add to the task
   */
  template <typename T,
            typename = std::enable_if_t<!std::is_same_v<std::decay_t<T>, Scalar> &&
                                        std::is_constructible_v<Scalar, T>>>
  void add_scalar_arg(T&& value);

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
  [[nodiscard]] std::string_view provenance() const;

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
  void add_communicator(std::string_view name);

  AutoTask() = LEGATE_DEFAULT_WHEN_CYTHON;

  AutoTask(AutoTask&&) noexcept            = default;
  AutoTask& operator=(AutoTask&&) noexcept = default;
  AutoTask(const AutoTask&)                = default;
  AutoTask& operator=(const AutoTask&)     = default;
  ~AutoTask() noexcept;

 private:
  friend class Runtime;
  explicit AutoTask(InternalSharedPtr<detail::AutoTask> impl);

  [[nodiscard]] const SharedPtr<detail::AutoTask>& impl_() const;
  [[nodiscard]] SharedPtr<detail::AutoTask> release_();
  [[nodiscard]] InternalSharedPtr<detail::LogicalArray> record_user_ref_(LogicalArray array);
  void clear_user_refs_();

  class Impl;
  InternalSharedPtr<Impl> pimpl_{};
};

/**
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
  void add_input(LogicalStore store);
  /**
   * @brief Adds a store partition to the task as input
   *
   * @param store_partition A store partition to add to the task as input
   * @param projection An optional symbolic point describing a mapping between points in the
   * launch domain and substores in the partition
   */
  void add_input(LogicalStorePartition store_partition,
                 std::optional<SymbolicPoint> projection = std::nullopt);
  /**
   * @brief Adds a store to the task as output
   *
   * The store will be unpartitioned but broadcasted to all the tasks
   *
   * @param store A store to add to the task as output
   */
  void add_output(LogicalStore store);
  /**
   * @brief Adds a store partition to the task as output
   *
   * @param store_partition A store partition to add to the task as output
   * @param projection An optional symbolic point describing a mapping between points in the
   * launch domain and substores in the partition
   */
  void add_output(LogicalStorePartition store_partition,
                  std::optional<SymbolicPoint> projection = std::nullopt);
  /**
   * @brief Adds a store to the task for reductions
   *
   * The store will be unpartitioned but broadcasted to all the tasks
   *
   * @param store A store to add to the task for reductions
   * @param redop_kind ID of the reduction operator to use. The store's type must support the
   * operator.
   */
  void add_reduction(LogicalStore store, ReductionOpKind redop_kind);
  /**
   * @brief Adds a store to the task for reductions
   *
   * The store will be unpartitioned but broadcasted to all the tasks
   *
   * @param store A store to add to the task for reductions
   * @param redop_kind ID of the reduction operator to use. The store's type must support the
   * operator.
   */
  void add_reduction(LogicalStore store, std::int32_t redop_kind);
  /**
   * @brief Adds a store partition to the task for reductions
   *
   * @param store_partition A store partition to add to the task for reductions
   * @param redop_kind ID of the reduction operator to use. The store's type must support the
   * operator.
   * @param projection An optional symbolic point describing a mapping between points in the
   * launch domain and substores in the partition
   */
  void add_reduction(LogicalStorePartition store_partition,
                     ReductionOpKind redop_kind,
                     std::optional<SymbolicPoint> projection = std::nullopt);
  /**
   * @brief Adds a store partition to the task for reductions
   *
   * @param store_partition A store partition to add to the task for reductions
   * @param redop_kind ID of the reduction operator to use. The store's type must support the
   * operator.
   * @param projection An optional symbolic point describing a mapping between points in the
   * launch domain and substores in the partition
   */
  void add_reduction(LogicalStorePartition store_partition,
                     std::int32_t redop_kind,
                     std::optional<SymbolicPoint> projection = std::nullopt);
  /**
   * @brief Adds a by-value scalar argument to the task
   *
   * @param scalar The Scalar to add to the task
   */
  void add_scalar_arg(const Scalar& scalar);
  /**
   * @brief Adds a by-value scalar argument to the task
   *
   * @tparam T The scalar value's type. Scalar must be constructible from a value of T
   * @param value The scalar value to convert to Scalar and add to the task
   */
  template <typename T,
            typename = std::enable_if_t<!std::is_same_v<std::decay_t<T>, Scalar> &&
                                        std::is_constructible_v<Scalar, T>>>
  void add_scalar_arg(T&& value);

  /**
   * @brief Returns the provenance information of this operation
   *
   * @return Provenance
   */
  [[nodiscard]] std::string_view provenance() const;

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
  void add_communicator(std::string_view name);

  ManualTask() = LEGATE_DEFAULT_WHEN_CYTHON;

  ManualTask(ManualTask&&) noexcept            = default;
  ManualTask& operator=(ManualTask&&) noexcept = default;
  ManualTask(const ManualTask&)                = default;
  ManualTask& operator=(const ManualTask&)     = default;
  ~ManualTask() noexcept;

 private:
  friend class Runtime;

  explicit ManualTask(InternalSharedPtr<detail::ManualTask> impl);

  [[nodiscard]] const SharedPtr<detail::ManualTask>& impl_() const;
  [[nodiscard]] SharedPtr<detail::ManualTask> release_();
  [[nodiscard]] InternalSharedPtr<detail::LogicalStore> record_user_ref_(LogicalStore store);
  [[nodiscard]] InternalSharedPtr<detail::LogicalStorePartition> record_user_ref_(
    LogicalStorePartition store_partition);
  void clear_user_refs_();

  class Impl;
  InternalSharedPtr<Impl> pimpl_{};
};

/** @} */

}  // namespace legate

#include <legate/operation/task.inl>
