/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/detail/task.h>
#include <legate/partitioning/detail/proxy/select.h>
#include <legate/partitioning/proxy.h>
#include <legate/utilities/abort.h>

namespace legate::detail {

inline std::variant<const TaskArrayArg*, Span<const TaskArrayArg>> ArgSelectVisitor::operator()(
  const ProxyArrayArgument& array) const
{
  switch (array.kind) {
    case ProxyArrayArgument::Kind::INPUT: return &task->inputs()[array.index];
    case ProxyArrayArgument::Kind::OUTPUT: return &task->outputs()[array.index];
    case ProxyArrayArgument::Kind::REDUCTION: return &task->reductions()[array.index];
  }
  // GCC is off its rocker, the switch fully covers the enum:
  //
  // /tmp/conda-croot/legate/work/src/cpp/legate/partitioning/detail/proxy/array.cc:28:1:
  // error: control reaches end of non-void function [-Werror=return-type]
  //     28 | }
  //        | ^
  LEGATE_ABORT("Unhandled array kind ", static_cast<std::uint8_t>(array.kind));
}

inline std::variant<const TaskArrayArg*, Span<const TaskArrayArg>> ArgSelectVisitor::operator()(
  const ProxyInputArguments&) const noexcept
{
  return task->inputs();
}

inline std::variant<const TaskArrayArg*, Span<const TaskArrayArg>> ArgSelectVisitor::operator()(
  const ProxyOutputArguments&) const noexcept
{
  return task->outputs();
}

inline std::variant<const TaskArrayArg*, Span<const TaskArrayArg>> ArgSelectVisitor::operator()(
  const ProxyReductionArguments&) const noexcept
{
  return task->reductions();
}

}  // namespace legate::detail
