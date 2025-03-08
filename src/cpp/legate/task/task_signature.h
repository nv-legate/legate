/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/doxygen.h>
#include <legate/utilities/shared_ptr.h>
#include <legate/utilities/span.h>

#include <cstdint>
#include <limits>
#include <optional>

/**
 * @file
 * @brief Class definition of legate::TaskSignature.
 */

namespace legate::detail {

class TaskSignature;

}  // namespace legate::detail

namespace legate {

class ProxyConstraint;

/**
 * @addtogroup task
 * @{
 */

/**
 * @brief A helper class for specifying a task's call signature.
 *
 * This class is used to statically declare a task's expected signature. For example:
 *
 * @snippet{trimleft} unit/task_signature/register.cc Example task signature
 *
 * A tasks signature describes how many input, output, scalar, or reduction arguments they
 * take, as well as any constraints that are to be applied to the task. If a task predeclares
 * its signature in this manner, the runtime will be able to perform a number of optimizations
 * and sanity-checks for the user, including (but not limited to):
 *
 * - Checking the number of arguments matches the expected signature (and raising exceptions if
 *   not).
 * - Automatically applying constraints on task arguments.
 * - Improved scheduling of tasks.
 *
 * @note While it is highly recommended that user statically declare their tasks' signatures,
 * the user is no longer allowed to deviate from the signature at runtime. For example, tasks
 * that predeclare their constraints are not allowed to add additional constraints during task
 * launch.
 */
class TaskSignature {
 public:
  /**
   * @brief Default-construct an empty TaskSignature.
   */
  TaskSignature();
  TaskSignature(const TaskSignature&)                = default;
  TaskSignature& operator=(const TaskSignature&)     = default;
  TaskSignature(TaskSignature&&) noexcept            = default;
  TaskSignature& operator=(TaskSignature&&) noexcept = default;
  ~TaskSignature();

  explicit TaskSignature(InternalSharedPtr<detail::TaskSignature> impl);

  /**
   * @brief A value indicating that a particular option has "unbounded" (or unknown) number of
   * possibilities.
   *
   * This is commonly used for e.g. `inputs()`, `outputs()`, `scalars()`, or `redops()` when a
   * task takes an unknown number of arguments, or when the upper limit on the number of
   * arguments is unknown.
   */
  static constexpr auto UNBOUNDED = std::numeric_limits<std::uint32_t>::max();

  /**
   * @brief Set the number of input arguments taken by the task.
   *
   * If `n` is `UNBOUNDED`, it signifies that the task takes a variable (or possibly unknown)
   * number of input arguments. Otherwise, this call signifies that a task takes exactly `n`
   * input arguments, no more, no less.
   *
   * @param n The argument specification.
   *
   * @return A reference to this.
   *
   * @see `inputs(std::uint32_t, std::uint32_t)`.
   */
  TaskSignature& inputs(std::uint32_t n) noexcept;

  /**
   * @brief Set the number of input arguments taken by the task.
   *
   * This call signifies that a task takes at least `low_bound` but no more than `upper_bound`
   * number of input arguments. If `upper_bound` is `UNBOUNDED`, then the task takes at least
   * `low_bound` number of arguments, but can take an unlimited number of arguments past that.
   *
   * If given, `upper_bound` must be strictly greater than `low_bound`.
   *
   * @param low_bound The lower bound on the number of input arguments.
   * @param upper_bound The upper bound on the number of input arguments.
   *
   * @return A reference to this.
   *
   * @throw std::out_of_range If `upper_bound` <= `low_bound`.
   *
   * @see `inputs(std::uint32_t)`.
   */
  TaskSignature& inputs(std::uint32_t low_bound, std::uint32_t upper_bound);

  /**
   * @brief Set the number of output arguments taken by the task.
   *
   * If `n` is `UNBOUNDED`, it signifies that the task takes a variable (or possibly unknown)
   * number of output arguments. Otherwise, this call signifies that a task takes exactly `n`
   * output arguments, no more, no less.
   *
   * @param n The argument specification.
   *
   * @return A reference to this.
   *
   * @see `outputs(std::uint32_t, std::uint32_t)`.
   */
  TaskSignature& outputs(std::uint32_t n) noexcept;

  /**
   * @brief Set the number of output arguments taken by the task.
   *
   * This call signifies that a task takes at least `low_bound` but no more than `upper_bound`
   * number of output arguments. If `upper_bound` is `UNBOUNDED`, then the task takes at least
   * `low_bound` number of arguments, but can take an unlimited number of arguments past that.
   *
   * If given, `upper_bound` must be strictly greater than `low_bound`.
   *
   * @param low_bound The lower bound on the number of output arguments.
   * @param upper_bound The upper bound on the number of output arguments.
   *
   * @return A reference to this.
   *
   * @throw std::out_of_range If `upper_bound` <= `low_bound`.
   *
   * @see `outputs(std::uint32_t)`.
   */
  TaskSignature& outputs(std::uint32_t low_bound, std::uint32_t upper_bound);

  /**
   * @brief Set the number of scalar arguments taken by the task.
   *
   * If `n` is `UNBOUNDED`, it signifies that the task takes a variable (or possibly unknown)
   * number of scalar arguments. Otherwise, this call signifies that a task takes exactly `n`
   * scalar arguments, no more, no less.
   *
   * @param n The argument specification.
   *
   * @return A reference to this.
   *
   * @see `scalars(std::uint32_t, std::uint32_t)`.
   */
  TaskSignature& scalars(std::uint32_t n) noexcept;

  /**
   * @brief Set the number of scalar arguments taken by the task.
   *
   * This call signifies that a task takes at least `low_bound` but no more than `upper_bound`
   * number of scalar arguments. If `upper_bound` is `UNBOUNDED`, then the task takes at least
   * `low_bound` number of arguments, but can take an unlimited number of arguments past that.
   *
   * If given, `upper_bound` must be strictly greater than `low_bound`.
   *
   * @param low_bound The lower bound on the number of scalar arguments.
   * @param upper_bound The upper bound on the number of scalar arguments.
   *
   * @return A reference to this.
   *
   * @throw std::out_of_range If `upper_bound` <= `low_bound`.
   *
   * @see `scalars(std::uint32_t)`.
   */
  TaskSignature& scalars(std::uint32_t low_bound, std::uint32_t upper_bound);

  /**
   * @brief Set the number of redop arguments taken by the task.
   *
   * If `n` is `UNBOUNDED`, it signifies that the task takes a variable (or possibly unknown)
   * number of redop arguments. Otherwise, this call signifies that a task takes exactly `n`
   * redop arguments, no more, no less.
   *
   * @param n The argument specification.
   *
   * @return A reference to this.
   *
   * @see `redops(std::uint32_t, std::uint32_t)`.
   */
  TaskSignature& redops(std::uint32_t n) noexcept;

  /**
   * @brief Set the number of redop arguments taken by the task.
   *
   * This call signifies that a task takes at least `low_bound` but no more than `upper_bound`
   * number of redop arguments. If `upper_bound` is `UNBOUNDED`, then the task takes at least
   * `low_bound` number of arguments, but can take an unlimited number of arguments past that.
   *
   * If given, `upper_bound` must be strictly greater than `low_bound`.
   *
   * @param low_bound The lower bound on the number of redop arguments.
   * @param upper_bound The upper bound on the number of redop arguments.
   *
   * @return A reference to this.
   *
   * @throw std::out_of_range If `upper_bound` <= `low_bound`.
   *
   * @see `redops(std::uint32_t)`.
   */
  TaskSignature& redops(std::uint32_t low_bound, std::uint32_t upper_bound);

  /**
   * @brief Set the constraints imposed on task arguments.
   *
   * Passing `std::nullopt` vs passing an empty range has different meanings:
   *
   * - If `std::nullopt` is passed, this is taken to mean that an unknown number of dynamic
   *   constraints (of that type) may be imposed on the task during launch.
   * - Passing an empty range signifies that exactly 0 constraints of the given type must be
   *   imposed on the task during launch.
   *
   * If any constraints are imposed via the use of this API (including empty ranges), tasks are
   * no longer allowed to add constraints dynamically during task construction.
   * `AutoTask::add_constraint()` will raise an exception in this case.
   *
   * @param constraints The constraints, or `std::nullopt` if none are imposed.
   *
   * @return A reference to this.
   */
  TaskSignature& constraints(std::optional<Span<const ProxyConstraint>> constraints);

  [[nodiscard]] const SharedPtr<detail::TaskSignature>& impl() const;

  friend bool operator==(const TaskSignature& lhs, const TaskSignature& rhs) noexcept;
  friend bool operator!=(const TaskSignature& lhs, const TaskSignature& rhs) noexcept;

 private:
  [[nodiscard]] SharedPtr<detail::TaskSignature>& impl_();

  SharedPtr<detail::TaskSignature> pimpl_{};
};

/** @} */

}  // namespace legate

#include <legate/task/task_signature.inl>
