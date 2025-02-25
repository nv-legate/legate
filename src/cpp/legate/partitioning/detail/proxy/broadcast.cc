
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

#include <legate/partitioning/detail/proxy/broadcast.h>

#include <legate/operation/detail/task.h>
#include <legate/partitioning/detail/constraint.h>
#include <legate/partitioning/detail/proxy/select.h>
#include <legate/partitioning/detail/proxy/validate.h>
#include <legate/partitioning/proxy.h>
#include <legate/utilities/tuple.h>

#include <cstdint>
#include <optional>
#include <tuple>

namespace legate::detail::proxy {

namespace {

void do_broadcast_final(const Variable* var,
                        const std::optional<tuple<std::uint32_t>>& axes,
                        AutoTask* task)
{
  auto&& constraint = [&](legate::Variable pub_var) {
    if (axes.has_value()) {
      return legate::broadcast(pub_var, *axes);
    }
    return legate::broadcast(pub_var);
  }(legate::Variable{var});

  task->add_constraint(constraint.impl(), /* bypass_signature_check */ true);
}

void do_broadcast(const TaskArrayArg* arg,
                  const std::optional<tuple<std::uint32_t>>& axes,
                  AutoTask* task)
{
  do_broadcast_final(task->find_or_declare_partition(arg->array), axes, task);
}

void do_broadcast(Span<const TaskArrayArg> args,
                  const std::optional<tuple<std::uint32_t>>& axes,
                  AutoTask* task)
{
  for (auto&& arg : args) {
    do_broadcast(&arg, axes, task);
  }
}

}  // namespace

Broadcast::Broadcast(value_type value, std::optional<tuple<std::uint32_t>> axes) noexcept
  : value_{std::move(value)}, axes_{std::move(axes)}
{
}

void Broadcast::validate(std::string_view task_name, const TaskSignature& signature) const
{
  std::visit(ValidateVisitor{task_name, signature, *this}, value());
}

void Broadcast::apply(AutoTask* task) const
{
  std::visit([&](const auto& arg) { do_broadcast(arg, axes(), task); },
             std::visit(ArgSelectVisitor{task}, value()));
}

bool Broadcast::operator==(const Constraint& rhs) const noexcept
{
  if (const auto* rhsptr = dynamic_cast<const Broadcast*>(&rhs)) {
    return std::tie(value(), axes()) == std::tie(rhsptr->value(), rhsptr->axes());
  }
  return false;
}

}  // namespace legate::detail::proxy
