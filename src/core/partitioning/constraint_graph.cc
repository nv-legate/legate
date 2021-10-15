/* Copyright 2021 NVIDIA Corporation
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

#include <sstream>

#include "legion.h"

#include "core/partitioning/constraint.h"
#include "core/partitioning/constraint_graph.h"

namespace legate {

extern Legion::Logger log_legate;

void ConstraintGraph::add_variable(std::shared_ptr<Variable> variable)
{
  variables_.push_back(std::move(variable));
}

void ConstraintGraph::add_constraint(std::shared_ptr<Constraint> constraint)
{
  constraints_.push_back(std::move(constraint));
}

void ConstraintGraph::join(const ConstraintGraph& other)
{
  auto& other_variables   = other.variables();
  auto& other_constraints = other.constraints();

  for (auto& other_variable : other_variables) variables_.push_back(other_variable);
  for (auto& other_constraint : other_constraints) constraints_.push_back(other_constraint);
}

void ConstraintGraph::dump()
{
  for (auto& constraint : constraints_) log_legate.debug("%s", constraint->to_string().c_str());
}

const std::vector<std::shared_ptr<Variable>>& ConstraintGraph::variables() const
{
  return variables_;
}

const std::vector<std::shared_ptr<Constraint>>& ConstraintGraph::constraints() const
{
  return constraints_;
}

}  // namespace legate
