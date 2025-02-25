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

#include <legate/operation/detail/task.h>
#include <legate/partitioning/detail/proxy/select.h>
#include <legate/partitioning/proxy.h>
#include <legate/utilities/abort.h>

namespace legate::detail::proxy {

inline std::variant<const TaskArrayArg*, Span<const TaskArrayArg>> ArgSelectVisitor::operator()(
  const legate::proxy::ArrayArgument& array) const
{
  switch (array.kind) {
    case legate::proxy::ArrayArgument::Kind::INPUT: return &task->inputs()[array.index];
    case legate::proxy::ArrayArgument::Kind::OUTPUT: return &task->outputs()[array.index];
    case legate::proxy::ArrayArgument::Kind::REDUCTION: return &task->reductions()[array.index];
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
  const legate::proxy::InputArguments&) const noexcept
{
  return task->inputs();
}

inline std::variant<const TaskArrayArg*, Span<const TaskArrayArg>> ArgSelectVisitor::operator()(
  const legate::proxy::OutputArguments&) const noexcept
{
  return task->outputs();
}

inline std::variant<const TaskArrayArg*, Span<const TaskArrayArg>> ArgSelectVisitor::operator()(
  const legate::proxy::ReductionArguments&) const noexcept
{
  return task->reductions();
}

}  // namespace legate::detail::proxy
