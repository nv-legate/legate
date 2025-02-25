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

#include <legate/partitioning/detail/proxy/bloat.h>

#include <legate/operation/detail/task.h>
#include <legate/partitioning/detail/constraint.h>
#include <legate/partitioning/detail/proxy/select.h>
#include <legate/partitioning/detail/proxy/validate.h>

#include <tuple>

namespace legate::detail::proxy {

namespace {

void do_bloat_final(const Variable* var_source,
                    const Variable* var_bloat,
                    const tuple<std::uint64_t>& low_offsets,
                    const tuple<std::uint64_t>& high_offsets,
                    AutoTask* task)
{
  if (var_source != var_bloat) {
    task->add_constraint(bloat(var_source, var_bloat, low_offsets, high_offsets),
                         /* bypass_signature_check */ true);
  }
}

void do_bloat(const TaskArrayArg* var_source,
              const TaskArrayArg* var_bloat,
              const tuple<std::uint64_t>& low_offsets,
              const tuple<std::uint64_t>& high_offsets,
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
              const tuple<std::uint64_t>& low_offsets,
              const tuple<std::uint64_t>& high_offsets,
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
              const tuple<std::uint64_t>& low_offsets,
              const tuple<std::uint64_t>& high_offsets,
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
              const tuple<std::uint64_t>& low_offsets,
              const tuple<std::uint64_t>& high_offsets,
              AutoTask* task)
{
  for (auto&& src : var_source) {
    do_bloat(&src, var_bloat, low_offsets, high_offsets, task);
  }
}

}  // namespace

Bloat::Bloat(value_type var_source,
             value_type var_bloat,
             tuple<std::uint64_t> low_offsets,
             tuple<std::uint64_t> high_offsets) noexcept
  : var_source_{std::move(var_source)},
    var_bloat_{std::move(var_bloat)},
    low_offsets_{std::move(low_offsets)},
    high_offsets_{std::move(high_offsets)}
{
}

void Bloat::validate(std::string_view task_name, const TaskSignature& signature) const
{
  const auto visitor = ValidateVisitor{task_name, signature, *this};

  std::visit(visitor, var_source());
  std::visit(visitor, var_bloat());
}

void Bloat::apply(AutoTask* task) const
{
  std::visit(
    [&](const auto& var_source, const auto& var_bloat) {
      do_bloat(var_source, var_bloat, low_offsets(), high_offsets(), task);
    },
    std::visit(ArgSelectVisitor{task}, var_source()),
    std::visit(ArgSelectVisitor{task}, var_bloat()));
}

bool Bloat::operator==(const Constraint& rhs) const noexcept
{
  if (const auto* rhsptr = dynamic_cast<const Bloat*>(&rhs)) {
    return std::tie(var_source(), var_bloat(), low_offsets(), high_offsets()) ==
           std::tie(rhsptr->var_source(),
                    rhsptr->var_bloat(),
                    rhsptr->low_offsets(),
                    rhsptr->high_offsets());
  }
  return false;
}

}  // namespace legate::detail::proxy
