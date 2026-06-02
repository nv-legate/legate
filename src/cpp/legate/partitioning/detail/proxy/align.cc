/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/proxy/align.h>

#include <legate/operation/detail/task.h>
#include <legate/operation/detail/task_store_arg.h>
#include <legate/partitioning/detail/constraint.h>
#include <legate/partitioning/detail/proxy/select.h>
#include <legate/partitioning/detail/proxy/validate.h>
#include <legate/utilities/detail/type_traits.h>

#include <tuple>

namespace legate::detail {

namespace {

void do_align_final(const Variable* left, const Variable* right, AutoTask* task)
{
  if (left != right) {
    task->add_constraint(align(left, right), /* bypass_signature_check */ true);
  }
}

void do_align(const TaskStoreArg* left, const TaskStoreArg* right, AutoTask* task)
{
  std::visit(Overload{[&](const InternalSharedPtr<LogicalStore>& left_st) {
                        std::visit(Overload{[&](const InternalSharedPtr<LogicalStore>& right_st) {
                                              do_align_final(
                                                task->find_or_declare_partition(left_st),
                                                task->find_or_declare_partition(right_st),
                                                task);
                                            },
                                            [&](const InternalSharedPtr<PhysicalStore>&) {
                                              // Do nothing for PhysicalStore
                                            }},
                                   right->store);
                      },
                      [&](const InternalSharedPtr<PhysicalStore>&) {
                        // Do nothing for PhysicalStore
                      }},
             left->store);
}

void do_align(const TaskStoreArg* left, Span<const TaskStoreArg> right, AutoTask* task)
{
  std::visit(Overload{[&](const InternalSharedPtr<LogicalStore>& left_st) {
                        const auto* left_part = task->find_or_declare_partition(left_st);

                        for (auto&& right_arg : right) {
                          std::visit(Overload{[&](const InternalSharedPtr<LogicalStore>& right_st) {
                                                do_align_final(
                                                  left_part,
                                                  task->find_or_declare_partition(right_st),
                                                  task);
                                              },
                                              [&](const InternalSharedPtr<PhysicalStore>&) {
                                                // Do nothing for PhysicalStore
                                              }},
                                     right_arg.store);
                        }
                      },
                      [&](const InternalSharedPtr<PhysicalStore>&) {
                        // Do nothing for PhysicalStore
                      }},
             left->store);
}

void do_align(Span<const TaskStoreArg> left, const TaskStoreArg* right, AutoTask* task)
{
  std::visit(Overload{[&](const InternalSharedPtr<LogicalStore>& right_st) {
                        const auto* right_part = task->find_or_declare_partition(right_st);

                        for (auto&& left_arg : left) {
                          std::visit(Overload{[&](const InternalSharedPtr<LogicalStore>& left_st) {
                                                do_align_final(
                                                  task->find_or_declare_partition(left_st),
                                                  right_part,
                                                  task);
                                              },
                                              [&](const InternalSharedPtr<PhysicalStore>&) {
                                                // Do nothing for PhysicalStore
                                              }},
                                     left_arg.store);
                        }
                      },
                      [&](const InternalSharedPtr<PhysicalStore>&) {
                        // Do nothing for PhysicalStore
                      }},
             right->store);
}

void do_align(Span<const TaskStoreArg> left, Span<const TaskStoreArg> right, AutoTask* task)
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
