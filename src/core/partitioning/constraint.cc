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

#include "core/data/scalar.h"
#include "core/partitioning/constraint.h"
#include "core/partitioning/partition.h"
#include "core/runtime/operation.h"

namespace legate {

Literal::Literal(const std::shared_ptr<Partition>& partition) : partition_(partition) {}

std::string Literal::to_string() const { return partition_->to_string(); }

void Literal::find_partition_symbols(std::vector<const Variable*>& partition_symbols) const {}

Variable::Variable(const Operation* op, int32_t id) : op_(op), id_(id) {}

bool operator==(const Variable& lhs, const Variable& rhs)
{
  return lhs.op_ == rhs.op_ && lhs.id_ == rhs.id_;
}

bool operator<(const Variable& lhs, const Variable& rhs) { return lhs.id_ < rhs.id_; }

void Variable::find_partition_symbols(std::vector<const Variable*>& partition_symbols) const
{
  partition_symbols.push_back(this);
}

std::string Variable::to_string() const
{
  std::stringstream ss;
  ss << "X" << id_ << "{" << op_->to_string() << "}";
  return std::move(ss).str();
}

Alignment::Alignment(std::unique_ptr<Expr>&& lhs, std::unique_ptr<Expr>&& rhs)
  : lhs_(std::forward<decltype(lhs_)>(lhs)), rhs_(std::forward<decltype(rhs_)>(rhs))
{
}

void Alignment::find_partition_symbols(std::vector<const Variable*>& partition_symbols) const
{
  lhs_->find_partition_symbols(partition_symbols);
  rhs_->find_partition_symbols(partition_symbols);
}

std::string Alignment::to_string() const
{
  std::stringstream ss;
  ss << "Align(" << lhs_->to_string() << ", " << rhs_->to_string() << ")";
  return std::move(ss).str();
}

Broadcast::Broadcast(std::unique_ptr<Variable> variable, tuple<int32_t>&& axes)
  : variable_(std::move(variable)), axes_(std::forward<tuple<int32_t>>(axes))
{
}

void Broadcast::find_partition_symbols(std::vector<const Variable*>& partition_symbols) const
{
  partition_symbols.push_back(variable_.get());
}

std::string Broadcast::to_string() const
{
  std::stringstream ss;
  ss << "Broadcast(" << variable_->to_string() << ", " << axes_.to_string() << ")";
  return std::move(ss).str();
}

std::unique_ptr<Alignment> align(const Variable* lhs, const Variable* rhs)
{
  // Since an Alignment object owns child nodes, inputs need to be copied
  return std::make_unique<Alignment>(std::make_unique<Variable>(*lhs),
                                     std::make_unique<Variable>(*rhs));
}

std::unique_ptr<Broadcast> broadcast(const Variable* variable, const tuple<int32_t>& axes)
{
  return broadcast(variable, tuple<int32_t>(axes));
}

std::unique_ptr<Broadcast> broadcast(const Variable* variable, tuple<int32_t>&& axes)
{
  return std::make_unique<Broadcast>(std::make_unique<Variable>(*variable),
                                     std::forward<tuple<int32_t>>(axes));
}

}  // namespace legate
