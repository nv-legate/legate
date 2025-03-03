/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/proxy/align.h>

#include <legate/operation/detail/task.h>
#include <legate/partitioning/detail/constraint.h>
#include <legate/partitioning/detail/proxy/select.h>
#include <legate/partitioning/detail/proxy/validate.h>

#include <tuple>

namespace legate::detail {

namespace {

void do_align_final(const Variable* left, const Variable* right, AutoTask* task)
{
  if (left != right) {
    task->add_constraint(align(left, right), /* bypass_signature_check */ true);
  }
}

void do_align(const TaskArrayArg* left, const TaskArrayArg* right, AutoTask* task)
{
  do_align_final(task->find_or_declare_partition(left->array),
                 task->find_or_declare_partition(right->array),
                 task);
}

void do_align(const TaskArrayArg* left, Span<const TaskArrayArg> right, AutoTask* task)
{
  const auto* left_part = task->find_or_declare_partition(left->array);

  for (auto&& right_arg : right) {
    do_align_final(left_part, task->find_or_declare_partition(right_arg.array), task);
  }
}

void do_align(Span<const TaskArrayArg> left, const TaskArrayArg* right, AutoTask* task)
{
  const auto* right_part = task->find_or_declare_partition(right->array);

  for (auto&& left_arg : left) {
    do_align_final(task->find_or_declare_partition(left_arg.array), right_part, task);
  }
}

void do_align(Span<const TaskArrayArg> left, Span<const TaskArrayArg> right, AutoTask* task)
{
  for (auto&& left_arg : left) {
    do_align(&left_arg, right, task);
  }
}

}  // namespace

ProxyAlign::ProxyAlign(value_type left, value_type right) noexcept
  : left_{std::move(left)}, right_{std::move(right)}
{
}

void ProxyAlign::validate(std::string_view task_name, const TaskSignature& signature) const
{
  const auto visitor = ValidateVisitor{task_name, signature, *this};

  std::visit(visitor, left());
  std::visit(visitor, right());
}

void ProxyAlign::apply(AutoTask* task) const
{
  std::visit([&](const auto& lhs, const auto& rhs) { do_align(lhs, rhs, task); },
             std::visit(ArgSelectVisitor{task}, left()),
             std::visit(ArgSelectVisitor{task}, right()));
}

bool ProxyAlign::operator==(const ProxyConstraint& rhs) const
{
  if (const auto* rhsptr = dynamic_cast<const ProxyAlign*>(&rhs)) {
    return std::tie(left(), right()) == std::tie(rhsptr->left(), rhsptr->right());
  }
  return false;
}

}  // namespace legate::detail
