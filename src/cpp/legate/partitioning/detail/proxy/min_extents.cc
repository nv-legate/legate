/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/proxy/min_extents.h>

#include <legate/operation/detail/task.h>
#include <legate/operation/detail/task_array_arg.h>
#include <legate/partitioning/constraint.h>
#include <legate/partitioning/detail/constraint.h>
#include <legate/partitioning/detail/proxy/select.h>
#include <legate/partitioning/detail/proxy/validate.h>
#include <legate/utilities/detail/type_traits.h>

#include <tuple>

namespace legate::detail {

namespace {

void do_min_extents_final(const Variable* variable,
                          const SmallVector<std::uint64_t, LEGATE_MAX_DIM>& minimum_extents,
                          AutoTask* task)
{
  task->add_constraint(legate::min_extents(legate::Variable{variable}, minimum_extents).impl(),
                       /* bypass_signature_check */ true);
}

void do_min_extents(const TaskArrayArg* variable,
                    const SmallVector<std::uint64_t, LEGATE_MAX_DIM>& min_extents,
                    AutoTask* task)
{
  std::visit(Overload{[&](const InternalSharedPtr<LogicalArray>& arr) {
                        do_min_extents_final(
                          task->find_or_declare_partition(arr), min_extents, task);
                      },
                      [&](const InternalSharedPtr<PhysicalArray>&) {
                        // Do nothing for PhysicalArray
                      }},
             variable->array);
}

void do_min_extents(Span<const TaskArrayArg> variables,
                    const SmallVector<std::uint64_t, LEGATE_MAX_DIM>& min_extents,
                    AutoTask* task)
{
  for (auto&& variable : variables) {
    do_min_extents(&variable, min_extents, task);
  }
}

}  // namespace

ProxyMinExtents::ProxyMinExtents(
  value_type variable, SmallVector<std::uint64_t, LEGATE_MAX_DIM> minimum_extents) noexcept
  : variable_{std::move(variable)}, minimum_extents_{std::move(minimum_extents)}
{
}

void ProxyMinExtents::validate(std::string_view task_name, const TaskSignature& signature) const
{
  const auto visitor = ValidateVisitor{task_name, signature, *this};

  std::visit(visitor, variable());
}

void ProxyMinExtents::apply(AutoTask* task) const
{
  std::visit([&](const auto& var) { do_min_extents(var, minimum_extents(), task); },
             std::visit(ArgSelectVisitor{task}, variable()));
}

bool ProxyMinExtents::operator==(const ProxyConstraint& rhs) const
{
  if (const auto* rhsptr = dynamic_cast<const ProxyMinExtents*>(&rhs)) {
    return std::tie(variable(), minimum_extents()) ==
           std::tie(rhsptr->variable(), rhsptr->minimum_extents());
  }
  return false;
}

}  // namespace legate::detail
