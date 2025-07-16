/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/constraint.h>

#include <legate/partitioning/detail/constraint.h>
#include <legate/partitioning/detail/proxy/align.h>
#include <legate/partitioning/detail/proxy/bloat.h>
#include <legate/partitioning/detail/proxy/broadcast.h>
#include <legate/partitioning/detail/proxy/image.h>
#include <legate/partitioning/detail/proxy/scale.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/shared_ptr.h>

namespace legate {

std::string Variable::to_string() const { return impl_->to_string(); }

std::string Constraint::to_string() const { return impl_->to_string(); }

Constraint::Constraint(InternalSharedPtr<detail::Constraint>&& impl) : impl_{std::move(impl)} {}

// ------------------------------------------------------------------------------------------

Constraint align(Variable lhs, Variable rhs)
{
  return Constraint{detail::align(lhs.impl(), rhs.impl())};
}

// ------------------------------------------------------------------------------------------

ProxyConstraint align(
  std::
    variant<ProxyArrayArgument, ProxyInputArguments, ProxyOutputArguments, ProxyReductionArguments>
      left,
  std::
    variant<ProxyArrayArgument, ProxyInputArguments, ProxyOutputArguments, ProxyReductionArguments>
      right)
{
  return ProxyConstraint{
    legate::make_shared<detail::ProxyAlign>(std::move(left), std::move(right))};
}

ProxyConstraint align(ProxyInputArguments proxies) { return align(proxies[0], proxies); }

ProxyConstraint align(ProxyOutputArguments proxies) { return align(proxies[0], proxies); }

// ------------------------------------------------------------------------------------------

Constraint broadcast(Variable variable) { return Constraint{detail::broadcast(variable.impl())}; }

Constraint broadcast(Variable variable, Span<const std::uint32_t> axes)
{
  return Constraint{
    detail::broadcast(variable.impl(), detail::SmallVector<std::uint32_t, LEGATE_MAX_DIM>{axes})};
}

// ------------------------------------------------------------------------------------------

ProxyConstraint broadcast(std::variant<ProxyArrayArgument,
                                       ProxyInputArguments,
                                       ProxyOutputArguments,
                                       ProxyReductionArguments> value,
                          const std::optional<tuple<std::uint32_t>>& axes)
{
  return ProxyConstraint{legate::make_shared<detail::ProxyBroadcast>(
    std::move(value),
    axes.has_value() ? std::make_optional<detail::SmallVector<std::uint32_t, LEGATE_MAX_DIM>>(
                         // &* to work around CCCL bug https://github.com/NVIDIA/cccl/issues/5116
                         detail::tags::iterator_tag,
                         &*(axes->begin()),
                         &*(axes->end()))
                     : std::nullopt)};
}

// ------------------------------------------------------------------------------------------

Constraint image(Variable var_function, Variable var_range, ImageComputationHint hint)
{
  return Constraint{detail::image(var_function.impl(), var_range.impl(), hint)};
}

// ------------------------------------------------------------------------------------------

ProxyConstraint image(
  std::
    variant<ProxyArrayArgument, ProxyInputArguments, ProxyOutputArguments, ProxyReductionArguments>
      var_function,
  std::
    variant<ProxyArrayArgument, ProxyInputArguments, ProxyOutputArguments, ProxyReductionArguments>
      var_range,
  std::optional<ImageComputationHint> hint)
{
  return ProxyConstraint{legate::make_shared<detail::ProxyImage>(
    std::move(var_function), std::move(var_range), std::move(hint))};
}

// ------------------------------------------------------------------------------------------

Constraint scale(Span<const std::uint64_t> factors, Variable var_smaller, Variable var_bigger)
{
  return Constraint{detail::scale(detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{factors},
                                  var_smaller.impl(),
                                  var_bigger.impl())};
}

// ------------------------------------------------------------------------------------------

ProxyConstraint scale(
  Span<const std::uint64_t> factors,
  std::
    variant<ProxyArrayArgument, ProxyInputArguments, ProxyOutputArguments, ProxyReductionArguments>
      var_smaller,
  std::
    variant<ProxyArrayArgument, ProxyInputArguments, ProxyOutputArguments, ProxyReductionArguments>
      var_bigger)
{
  return ProxyConstraint{legate::make_shared<detail::ProxyScale>(
    detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{factors},
    std::move(var_smaller),
    std::move(var_bigger))};
}

// ------------------------------------------------------------------------------------------

Constraint bloat(Variable var_source,
                 Variable var_bloat,
                 Span<const std::uint64_t> low_offsets,
                 Span<const std::uint64_t> high_offsets)
{
  return Constraint{
    detail::bloat(var_source.impl(),
                  var_bloat.impl(),
                  detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{low_offsets},
                  detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{high_offsets})};
}

// ------------------------------------------------------------------------------------------

ProxyConstraint bloat(
  std::
    variant<ProxyArrayArgument, ProxyInputArguments, ProxyOutputArguments, ProxyReductionArguments>
      var_source,
  std::
    variant<ProxyArrayArgument, ProxyInputArguments, ProxyOutputArguments, ProxyReductionArguments>
      var_bloat,
  Span<const std::uint64_t> low_offsets,
  Span<const std::uint64_t> high_offsets)
{
  return ProxyConstraint{legate::make_shared<detail::ProxyBloat>(
    std::move(var_source),
    std::move(var_bloat),
    detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{low_offsets},
    detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{high_offsets})};
}

}  // namespace legate
