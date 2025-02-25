/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <legate/utilities/span.h>

#include <variant>

namespace legate::proxy {

class ArrayArgument;
class InputArguments;
class OutputArguments;
class ReductionArguments;

}  // namespace legate::proxy

namespace legate::detail {

class TaskArrayArg;
class AutoTask;

}  // namespace legate::detail

namespace legate::detail::proxy {

/**
 * @brief A visitor used to select a particular argument, or argument group from a task.
 */
class ArgSelectVisitor {
 public:
  /**
   * @brief Selection overload for Arrays.
   *
   * @param array The array.
   *
   * @return Always returns a `TaskArrayArg`, i.e. a specific argument of the task.
   */
  [[nodiscard]] std::variant<const TaskArrayArg*, Span<const TaskArrayArg>> operator()(
    const legate::proxy::ArrayArgument& array) const;

  /**
   * @brief Selection overload for inputs.
   *
   * @return Always returns `task->inputs()`.
   */
  [[nodiscard]] std::variant<const TaskArrayArg*, Span<const TaskArrayArg>> operator()(
    const legate::proxy::InputArguments&) const noexcept;

  /**
   * @brief Selection overload for outputs.
   *
   * @return Always returns `task->outputs()`.
   */
  [[nodiscard]] std::variant<const TaskArrayArg*, Span<const TaskArrayArg>> operator()(
    const legate::proxy::OutputArguments&) const noexcept;

  /**
   * @brief Selection overload for reductions.
   *
   * @return Always returns `task->reductions()`.
   */
  [[nodiscard]] std::variant<const TaskArrayArg*, Span<const TaskArrayArg>> operator()(
    const legate::proxy::ReductionArguments&) const noexcept;

  const AutoTask* task{}; /** The task to select from */
};

}  // namespace legate::detail::proxy

#include <legate/partitioning/detail/proxy/select.inl>
