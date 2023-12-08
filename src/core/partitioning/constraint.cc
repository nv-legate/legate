/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "core/partitioning/constraint.h"

#include "core/partitioning/detail/constraint.h"

namespace legate {

std::string Variable::to_string() const { return impl_->to_string(); }

std::string Constraint::to_string() const { return impl_->to_string(); }

Constraint::Constraint(InternalSharedPtr<detail::Constraint>&& impl) : impl_{std::move(impl)} {}

Constraint align(Variable lhs, Variable rhs)
{
  return Constraint{detail::align(lhs.impl(), rhs.impl())};
}

Constraint broadcast(Variable variable) { return Constraint{detail::broadcast(variable.impl())}; }

Constraint broadcast(Variable variable, const tuple<int32_t>& axes)
{
  return Constraint{detail::broadcast(variable.impl(), tuple<int32_t>(axes))};
}

Constraint image(Variable var_function, Variable var_range)
{
  return Constraint{detail::image(var_function.impl(), var_range.impl())};
}

Constraint scale(const Shape& factors, Variable var_smaller, Variable var_bigger)
{
  return Constraint{detail::scale(factors, var_smaller.impl(), var_bigger.impl())};
}

Constraint bloat(Variable var_source,
                 Variable var_bloat,
                 const Shape& low_offsets,
                 const Shape& high_offsets)
{
  return Constraint{detail::bloat(var_source.impl(), var_bloat.impl(), low_offsets, high_offsets)};
}

}  // namespace legate
