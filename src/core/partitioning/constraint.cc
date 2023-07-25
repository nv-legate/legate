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

Variable::Variable(const detail::Variable* impl) : impl_(impl) {}

Variable::~Variable() {}

Variable::Variable(const Variable&) = default;

Variable& Variable::operator=(const Variable&) = default;

std::string Constraint::to_string() const { return impl_->to_string(); }

Constraint::Constraint(detail::Constraint* impl) : impl_(impl) {}

Constraint::~Constraint() { delete impl_; }

Constraint::Constraint(Constraint&& other) : impl_(other.impl_) { other.impl_ = nullptr; }

Constraint& Constraint::operator=(Constraint&& other)
{
  impl_       = other.impl_;
  other.impl_ = nullptr;
  return *this;
}

detail::Constraint* Constraint::release()
{
  auto result = impl_;
  impl_       = nullptr;
  return result;
}

Constraint align(Variable lhs, Variable rhs)
{
  return Constraint(detail::align(lhs.impl(), rhs.impl()).release());
}

Constraint broadcast(Variable variable, const tuple<int32_t>& axes)
{
  return Constraint(detail::broadcast(variable.impl(), tuple<int32_t>(axes)).release());
}

Constraint image(Variable var_function, Variable var_range)
{
  return Constraint(detail::image(var_function.impl(), var_range.impl()).release());
}

}  // namespace legate
