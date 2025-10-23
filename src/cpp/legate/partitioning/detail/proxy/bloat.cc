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
#include <legate/utilities/detail/type_traits.h>

#include <algorithm>
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
  std::visit(Overload{[&](const InternalSharedPtr<LogicalArray>& source_arr) {
                        std::visit(Overload{[&](const InternalSharedPtr<LogicalArray>& bloat_arr) {
                                              do_bloat_final(
                                                task->find_or_declare_partition(source_arr),
                                                task->find_or_declare_partition(bloat_arr),
                                                low_offsets,
                                                high_offsets,
                                                task);
                                            },
                                            [&](const InternalSharedPtr<PhysicalArray>&) {
                                              // Do nothing for PhysicalArray
                                            }},
                                   var_bloat->array);
                      },
                      [&](const InternalSharedPtr<PhysicalArray>&) {
                        // Do nothing for PhysicalArray
                      }},
             var_source->array);
}

void do_bloat(const TaskArrayArg* var_source,
              Span<const TaskArrayArg> var_bloat,
              Span<const std::uint64_t> low_offsets,
              Span<const std::uint64_t> high_offsets,
              AutoTask* task)
{
  std::visit(Overload{[&](const InternalSharedPtr<LogicalArray>& source_arr) {
                        const auto* source_part = task->find_or_declare_partition(source_arr);

                        for (auto&& bloat : var_bloat) {
                          std::visit(
                            Overload{[&](const InternalSharedPtr<LogicalArray>& bloat_arr) {
                                       do_bloat_final(source_part,
                                                      task->find_or_declare_partition(bloat_arr),
                                                      low_offsets,
                                                      high_offsets,
                                                      task);
                                     },
                                     [&](const InternalSharedPtr<PhysicalArray>&) {
                                       // Do nothing for PhysicalArray
                                     }},
                            bloat.array);
                        }
                      },
                      [&](const InternalSharedPtr<PhysicalArray>&) {
                        // Do nothing for PhysicalArray
                      }},
             var_source->array);
}

void do_bloat(Span<const TaskArrayArg> var_source,
              const TaskArrayArg* var_bloat,
              Span<const std::uint64_t> low_offsets,
              Span<const std::uint64_t> high_offsets,
              AutoTask* task)
{
  std::visit(Overload{[&](const InternalSharedPtr<LogicalArray>& bloat_arr) {
                        const auto* bloat_part = task->find_or_declare_partition(bloat_arr);

                        for (auto&& src : var_source) {
                          std::visit(
                            Overload{[&](const InternalSharedPtr<LogicalArray>& source_arr) {
                                       do_bloat_final(task->find_or_declare_partition(source_arr),
                                                      bloat_part,
                                                      low_offsets,
                                                      high_offsets,
                                                      task);
                                     },
                                     [&](const InternalSharedPtr<PhysicalArray>&) {
                                       // Do nothing for PhysicalArray
                                     }},
                            src.array);
                        }
                      },
                      [&](const InternalSharedPtr<PhysicalArray>&) {
                        // Do nothing for PhysicalArray
                      }},
             var_bloat->array);
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
           std::equal(low_offsets().begin(),
                      low_offsets().end(),
                      rhsptr->low_offsets().begin(),
                      rhsptr->low_offsets().end()) &&
           std::equal(high_offsets().begin(),
                      high_offsets().end(),
                      rhsptr->high_offsets().begin(),
                      rhsptr->high_offsets().end());
  }
  return false;
}

}  // namespace legate::detail
