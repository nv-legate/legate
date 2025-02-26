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

#include <legate/partitioning/detail/proxy/image.h>

#include <legate/operation/detail/task.h>
#include <legate/partitioning/constraint.h>
#include <legate/partitioning/detail/constraint.h>
#include <legate/partitioning/detail/proxy/select.h>
#include <legate/partitioning/detail/proxy/validate.h>

#include <tuple>

namespace legate::detail {

namespace {

void do_image_final(const Variable* var_function,
                    const Variable* var_range,
                    const std::optional<ImageComputationHint>& hint,
                    AutoTask* task)
{
  if (var_function == var_range) {
    return;
  }

  auto&& constraint = [&](legate::Variable pub_var_function, legate::Variable pub_var_range) {
    if (hint.has_value()) {
      return legate::image(pub_var_function, pub_var_range, *hint);
    }
    return legate::image(pub_var_function, pub_var_range);
  }(legate::Variable{var_function}, legate::Variable{var_range});

  task->add_constraint(constraint.impl(), /* bypass_signature_check */ true);
}

void do_image(const TaskArrayArg* var_function,
              const TaskArrayArg* var_range,
              const std::optional<ImageComputationHint>& hint,
              AutoTask* task)
{
  do_image_final(task->find_or_declare_partition(var_function->array),
                 task->find_or_declare_partition(var_range->array),
                 hint,
                 task);
}

void do_image(const TaskArrayArg* var_function,
              Span<const TaskArrayArg> var_ranges,
              const std::optional<ImageComputationHint>& hint,
              AutoTask* task)
{
  const auto* func_part = task->find_or_declare_partition(var_function->array);

  for (auto&& range : var_ranges) {
    do_image_final(func_part, task->find_or_declare_partition(range.array), hint, task);
  }
}

void do_image(Span<const TaskArrayArg> var_functions,
              const TaskArrayArg* var_range,
              const std::optional<ImageComputationHint>& hint,
              AutoTask* task)
{
  const auto* range_part = task->find_or_declare_partition(var_range->array);

  for (auto&& func : var_functions) {
    do_image_final(task->find_or_declare_partition(func.array), range_part, hint, task);
  }
}

void do_image(Span<const TaskArrayArg> var_functions,
              Span<const TaskArrayArg> var_ranges,
              const std::optional<ImageComputationHint>& hint,
              AutoTask* task)
{
  for (auto&& func : var_functions) {
    do_image(&func, var_ranges, hint, task);
  }
}

}  // namespace

ProxyImage::ProxyImage(value_type var_function,
                       value_type var_range,
                       std::optional<ImageComputationHint> hint) noexcept
  : var_function_{std::move(var_function)}, var_range_{std::move(var_range)}, hint_{std::move(hint)}
{
}

void ProxyImage::validate(std::string_view task_name, const TaskSignature& signature) const
{
  const auto visitor = ValidateVisitor{task_name, signature, *this};

  std::visit(visitor, var_function());
  std::visit(visitor, var_range());
}

void ProxyImage::apply(AutoTask* task) const
{
  std::visit([&](const auto& var_function,
                 const auto& var_range) { do_image(var_function, var_range, hint(), task); },
             std::visit(ArgSelectVisitor{task}, var_function()),
             std::visit(ArgSelectVisitor{task}, var_range()));
}

bool ProxyImage::operator==(const ProxyConstraint& rhs) const
{
  if (const auto* rhsptr = dynamic_cast<const ProxyImage*>(&rhs)) {
    return std::tie(var_function(), var_range(), hint()) ==
           std::tie(rhsptr->var_function(), rhsptr->var_range(), rhsptr->hint());
  }
  return false;
}

}  // namespace legate::detail
