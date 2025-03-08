/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/task/detail/task_signature.h>

#include <legate/operation/detail/task.h>
#include <legate/task/task_signature.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/detail/type_traits.h>
#include <legate/utilities/detail/zip.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
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

bool operator==(const TaskSignature::Nargs& lhs, const TaskSignature::Nargs& rhs) noexcept
{
  // To silence
  //
  // /path/to//src/cpp/legate/task/detail/task_signature.cc:88:6:
  // error: an exception may be thrown in function 'operator==' which should not throw
  // exceptions [bugprone-exception-escape,-warnings-as-errors]
  // 88 | bool operator==(const TaskSignature::Nargs& lhs, const TaskSignature::Nargs& rhs) noexcept
  //    |      ^
  //
  // This is due to std::variant operator== not having a dynamic noexcept specification. None
  // of the code below could ever possibly throw an exception.
  try {
    return lhs.value() == rhs.value();
  } catch (...) {
    return false;
  }
}

bool operator!=(const TaskSignature::Nargs& lhs, const TaskSignature::Nargs& rhs) noexcept
{
  return !(lhs == rhs);
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

bool operator==(const TaskSignature& lhs, const TaskSignature& rhs) noexcept
{
  // Fast path
  if (std::addressof(lhs) == std::addressof(rhs)) {
    return true;
  }

  if (std::tie(lhs.inputs(), lhs.outputs(), lhs.scalars(), lhs.redops()) !=
      std::tie(rhs.inputs(), rhs.outputs(), rhs.scalars(), rhs.redops())) {
    return false;
  }

  const auto& lhs_csts = lhs.constraints();
  const auto& rhs_csts = rhs.constraints();

  if (lhs_csts.has_value() != rhs_csts.has_value()) {
    return false;
  }
  // At this point, either both have a value, or neither has a value. If neither has a value,
  // then we are done
  if (!lhs_csts.has_value()) {
    return true;
  }

  const auto& lhs_span = *lhs_csts;
  const auto& rhs_span = *rhs_csts;

  if (lhs_span.size() != rhs_span.size()) {
    return false;
  }

  auto&& zipper = zip_equal(lhs_span, rhs_span);

  return std::all_of(zipper.begin(), zipper.end(), [](const auto& zip_it) {
    const auto& [lhs_val, rhs_val] = zip_it;

    return *lhs_val == *rhs_val;
  });
}

bool operator!=(const TaskSignature& lhs, const TaskSignature& rhs) noexcept
{
  return !(lhs == rhs);
}

}  // namespace legate::detail
