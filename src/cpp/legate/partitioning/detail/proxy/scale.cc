/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/proxy/scale.h>

#include <legate/operation/detail/task.h>
#include <legate/partitioning/detail/constraint.h>
#include <legate/partitioning/detail/proxy/select.h>
#include <legate/partitioning/detail/proxy/validate.h>

#include <algorithm>
#include <tuple>

namespace legate::detail {

namespace {

void do_scale_final(Span<const std::uint64_t> factors,
                    const Variable* var_smaller,
                    const Variable* var_bigger,
                    AutoTask* task)
{
  if (var_smaller != var_bigger) {
    task->add_constraint(
      scale(SmallVector<std::uint64_t, LEGATE_MAX_DIM>{factors}, var_smaller, var_bigger),
      /* bypass_signature_check */ true);
  }
}

void do_scale(Span<const std::uint64_t> factors,
              const TaskArrayArg* var_smaller,
              const TaskArrayArg* var_bigger,
              AutoTask* task)
{
  do_scale_final(factors,
                 task->find_or_declare_partition(var_smaller->array),
                 task->find_or_declare_partition(var_bigger->array),
                 task);
}

void do_scale(Span<const std::uint64_t> factors,
              const TaskArrayArg* var_smaller,
              Span<const TaskArrayArg> var_bigger,
              AutoTask* task)
{
  const auto* small_part = task->find_or_declare_partition(var_smaller->array);

  for (auto&& big : var_bigger) {
    do_scale_final(factors, small_part, task->find_or_declare_partition(big.array), task);
  }
}

void do_scale(Span<const std::uint64_t> factors,
              Span<const TaskArrayArg> var_smaller,
              const TaskArrayArg* var_bigger,
              AutoTask* task)
{
  const auto* big_part = task->find_or_declare_partition(var_bigger->array);

  for (auto&& small : var_smaller) {
    do_scale_final(factors, task->find_or_declare_partition(small.array), big_part, task);
  }
}

void do_scale(Span<const std::uint64_t> factors,
              Span<const TaskArrayArg> var_smaller,
              Span<const TaskArrayArg> var_bigger,
              AutoTask* task)
{
  for (auto&& small : var_smaller) {
    do_scale(factors, &small, var_bigger, task);
  }
}

}  // namespace

ProxyScale::ProxyScale(SmallVector<std::uint64_t, LEGATE_MAX_DIM> factors,
                       value_type var_smaller,
                       value_type var_bigger) noexcept
  : factors_{std::move(factors)},
    var_smaller_{std::move(var_smaller)},
    var_bigger_{std::move(var_bigger)}
{
}

void ProxyScale::validate(std::string_view task_name, const TaskSignature& signature) const
{
  const auto visitor = ValidateVisitor{task_name, signature, *this};

  std::visit(visitor, var_smaller());
  std::visit(visitor, var_bigger());
}

void ProxyScale::apply(AutoTask* task) const
{
  std::visit([&](const auto& var_smaller,
                 const auto& var_bigger) { do_scale(factors(), var_smaller, var_bigger, task); },
             std::visit(ArgSelectVisitor{task}, var_smaller()),
             std::visit(ArgSelectVisitor{task}, var_bigger()));
}

bool ProxyScale::operator==(const ProxyConstraint& rhs) const
{
  if (const auto* rhsptr = dynamic_cast<const ProxyScale*>(&rhs)) {
    return std::tie(var_smaller(), var_bigger()) ==
             std::tie(rhsptr->var_smaller(), rhsptr->var_bigger()) &&
           std::equal(factors().begin(),
                      factors().end(),
                      rhsptr->factors().begin(),
                      rhsptr->factors().end());
  }
  return false;
}

}  // namespace legate::detail
