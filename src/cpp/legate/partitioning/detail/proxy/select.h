/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/span.h>

#include <variant>

namespace legate {

class ProxyArrayArgument;
class ProxyInputArguments;
class ProxyOutputArguments;
class ProxyReductionArguments;

}  // namespace legate

namespace legate::detail {

class TaskArrayArg;

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
    const ProxyArrayArgument& array) const;

  /**
   * @brief Selection overload for inputs.
   *
   * @return Always returns `task->inputs()`.
   */
  [[nodiscard]] std::variant<const TaskArrayArg*, Span<const TaskArrayArg>> operator()(
    const ProxyInputArguments&) const noexcept;

  /**
   * @brief Selection overload for outputs.
   *
   * @return Always returns `task->outputs()`.
   */
  [[nodiscard]] std::variant<const TaskArrayArg*, Span<const TaskArrayArg>> operator()(
    const ProxyOutputArguments&) const noexcept;

  /**
   * @brief Selection overload for reductions.
   *
   * @return Always returns `task->reductions()`.
   */
  [[nodiscard]] std::variant<const TaskArrayArg*, Span<const TaskArrayArg>> operator()(
    const ProxyReductionArguments&) const noexcept;

  const AutoTask* task{}; /** The task to select from */
};

}  // namespace legate::detail

#include <legate/partitioning/detail/proxy/select.inl>
