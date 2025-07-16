/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/proxy/bloat.h>

#include <legate/operation/detail/task.h>
#include <legate/partitioning/detail/constraint.h>
#include <legate/partitioning/detail/proxy/select.h>
#include <legate/partitioning/detail/proxy/validate.h>

#include <tuple>

namespace legate::detail {

namespace {

void do_bloat_final(const Variable* var_source,
                    const Variable* var_bloat,
                    Span<const std::uint64_t> low_offsets,
                    Span<const std::uint64_t> high_offsets,
                    AutoTask* task)
{
  if (var_source != var_bloat) {
    task->add_constraint(bloat(var_source,
                               var_bloat,
                               SmallVector<std::uint64_t, LEGATE_MAX_DIM>{low_offsets},
                               SmallVector<std::uint64_t, LEGATE_MAX_DIM>{high_offsets}),
                         /* bypass_signature_check */ true);
  }
}

void do_bloat(const TaskArrayArg* var_source,
              const TaskArrayArg* var_bloat,
              Span<const std::uint64_t> low_offsets,
              Span<const std::uint64_t> high_offsets,
              AutoTask* task)
{
  do_bloat_final(task->find_or_declare_partition(var_source->array),
                 task->find_or_declare_partition(var_bloat->array),
                 low_offsets,
                 high_offsets,
                 task);
}

void do_bloat(const TaskArrayArg* var_source,
              Span<const TaskArrayArg> var_bloat,
              Span<const std::uint64_t> low_offsets,
              Span<const std::uint64_t> high_offsets,
              AutoTask* task)
{
  const auto* source_part = task->find_or_declare_partition(var_source->array);

  for (auto&& bloat : var_bloat) {
    do_bloat_final(
      source_part, task->find_or_declare_partition(bloat.array), low_offsets, high_offsets, task);
  }
}

void do_bloat(Span<const TaskArrayArg> var_source,
              const TaskArrayArg* var_bloat,
              Span<const std::uint64_t> low_offsets,
              Span<const std::uint64_t> high_offsets,
              AutoTask* task)
{
  const auto* bloat_part = task->find_or_declare_partition(var_bloat->array);

  for (auto&& src : var_source) {
    do_bloat_final(
      task->find_or_declare_partition(src.array), bloat_part, low_offsets, high_offsets, task);
  }
}

void do_bloat(Span<const TaskArrayArg> var_source,
              Span<const TaskArrayArg> var_bloat,
              Span<const std::uint64_t> low_offsets,
              Span<const std::uint64_t> high_offsets,
              AutoTask* task)
{
  for (auto&& src : var_source) {
    do_bloat(&src, var_bloat, low_offsets, high_offsets, task);
  }
}

}  // namespace

ProxyBloat::ProxyBloat(value_type var_source,
                       value_type var_bloat,
                       SmallVector<std::uint64_t, LEGATE_MAX_DIM> low_offsets,
                       SmallVector<std::uint64_t, LEGATE_MAX_DIM> high_offsets) noexcept
  : var_source_{std::move(var_source)},
    var_bloat_{std::move(var_bloat)},
    low_offsets_{std::move(low_offsets)},
    high_offsets_{std::move(high_offsets)}
{
}

void ProxyBloat::validate(std::string_view task_name, const TaskSignature& signature) const
{
  const auto visitor = ValidateVisitor{task_name, signature, *this};

  std::visit(visitor, var_source());
  std::visit(visitor, var_bloat());
}

void ProxyBloat::apply(AutoTask* task) const
{
  std::visit(
    [&](const auto& var_source, const auto& var_bloat) {
      do_bloat(var_source, var_bloat, low_offsets(), high_offsets(), task);
    },
    std::visit(ArgSelectVisitor{task}, var_source()),
    std::visit(ArgSelectVisitor{task}, var_bloat()));
}

bool ProxyBloat::operator==(const ProxyConstraint& rhs) const
{
  if (const auto* rhsptr = dynamic_cast<const ProxyBloat*>(&rhs)) {
    return std::tie(var_source(), var_bloat()) ==
             std::tie(rhsptr->var_source(), rhsptr->var_bloat()) &&
           low_offsets().deep_equal(rhsptr->low_offsets()) &&
           high_offsets().deep_equal(rhsptr->high_offsets());
  }
  return false;
}

}  // namespace legate::detail
