/* Copyright 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
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
