
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
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

namespace legate::detail {

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

ProxyBroadcast::ProxyBroadcast(value_type value, std::optional<tuple<std::uint32_t>> axes) noexcept
  : value_{std::move(value)}, axes_{std::move(axes)}
{
}

void ProxyBroadcast::validate(std::string_view task_name, const TaskSignature& signature) const
{
  std::visit(ValidateVisitor{task_name, signature, *this}, value());
}

void ProxyBroadcast::apply(AutoTask* task) const
{
  std::visit([&](const auto& arg) { do_broadcast(arg, axes(), task); },
             std::visit(ArgSelectVisitor{task}, value()));
}

bool ProxyBroadcast::operator==(const ProxyConstraint& rhs) const
{
  if (const auto* rhsptr = dynamic_cast<const ProxyBroadcast*>(&rhs)) {
    return std::tie(value(), axes()) == std::tie(rhsptr->value(), rhsptr->axes());
  }
  return false;
}

}  // namespace legate::detail
