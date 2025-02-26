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

#include <legate/task/detail/task_signature.h>

#include <legate/operation/detail/task.h>
#include <legate/task/task_signature.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/detail/type_traits.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fmt {

format_context::iterator formatter<legate::detail::TaskSignature::Nargs>::format(
  const legate::detail::TaskSignature::Nargs& nargs, format_context& ctx) const
{
  constexpr auto visitor =
    legate::detail::Overload{[](std::uint32_t v) { return fmt::format("Nargs({})", v); },
                             [](std::pair<std::uint32_t, std::uint32_t> pair) {
                               std::string ret;

                               fmt::format_to(std::back_inserter(ret), "Nargs({}, ", pair.first);
                               if (pair.second == legate::TaskSignature::UNBOUNDED) {
                                 ret += "unbounded";
                               } else {
                                 fmt::format_to(std::back_inserter(ret), "{}", pair.second);
                               }
                               ret += ')';
                               return ret;
                             }};

  return formatter<std::string>::format(std::visit(visitor, nargs.value()), ctx);
}

}  // namespace fmt

namespace legate::detail {

// ------------------------------------------------------------------------------------------

TaskSignature::Nargs::Nargs(std::uint32_t value) : value_{value} {}

TaskSignature::Nargs::Nargs(std::uint32_t lower, std::uint32_t upper)
  : value_{std::make_pair(lower, upper)}
{
  if (lower >= upper) {
    throw TracedException<std::out_of_range>{fmt::format(
      "Invalid argument range: {}, upper bound must be strictly greater than the lower bound",
      std::make_pair(lower, upper))};
  }
}

std::uint32_t TaskSignature::Nargs::upper_limit() const
{
  return std::visit(Overload{[](std::uint32_t v) { return v; },
                             [](std::pair<std::uint32_t, std::uint32_t> pair) {
                               return pair.second == legate::TaskSignature::UNBOUNDED ? pair.first
                                                                                      : pair.second;
                             }},
                    value());
}

bool TaskSignature::Nargs::compatible_with(std::size_t size, bool strict) const
{
  return std::visit(Overload{[=](std::uint32_t v) { return strict ? size == v : size <= v; },
                             [=](std::pair<std::uint32_t, std::uint32_t> pair) {
                               return (pair.first <= size) && (size <= pair.second);
                             }},
                    value());
}

// ------------------------------------------------------------------------------------------

void TaskSignature::constraints(
  std::optional<std::vector<InternalSharedPtr<detail::ProxyConstraint>>> cstrnts) noexcept
{
  constraints_ = std::move(cstrnts);
}

void TaskSignature::validate(std::string_view task_name) const
{
  if (const auto& csts = constraints(); csts.has_value()) {
    for (auto&& c : *csts) {
      c->validate(task_name, *this);
    }
  }
}

void TaskSignature::check_signature(const Task& task) const
{
  constexpr auto check_arg_sizes = [](const std::optional<Nargs>& sig_value,
                                      std::size_t task_arg_size,
                                      std::string_view arg_type) {
    if (sig_value.has_value() && !sig_value->compatible_with(task_arg_size)) {
      throw TracedException<std::out_of_range>{
        fmt::format("Invalid arguments to task. Expected {} {} arguments, have {}",
                    *sig_value,
                    arg_type,
                    task_arg_size)};
    }
  };

  check_arg_sizes(inputs(), task.inputs().size(), "input");
  check_arg_sizes(outputs(), task.outputs().size(), "output");
  check_arg_sizes(redops(), task.reductions().size(), "reduction");
  check_arg_sizes(scalars(), task.scalars().size(), "scalar");
}

void TaskSignature::apply_constraints(AutoTask* task) const
{
  if (const auto& csts = constraints(); csts.has_value()) {
    for (auto&& c : *csts) {
      c->apply(task);
    }
  }
}

}  // namespace legate::detail
