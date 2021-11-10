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

Variable::Variable(const Operation* op, int32_t id) : op_(op), id_(id) {}

bool operator<(const Variable& lhs, const Variable& rhs)
{
  if (lhs.op_ > rhs.op_)
    return false;
  else if (lhs.op_ < rhs.op_)
    return true;
  return lhs.id_ < rhs.id_;
}

std::string Variable::to_string() const
{
  std::stringstream ss;
  ss << "X" << id_ << "{" << op_->to_string() << "}";
  return ss.str();
}

struct Alignment : public Constraint {
 public:
  Alignment(std::shared_ptr<Expr> lhs, std::shared_ptr<Expr> rhs);

 public:
  virtual std::string to_string() const override;

 private:
  std::shared_ptr<Expr> lhs_;
  std::shared_ptr<Expr> rhs_;
};

Alignment::Alignment(std::shared_ptr<Expr> lhs, std::shared_ptr<Expr> rhs)
  : lhs_(std::move(lhs)), rhs_(std::move(rhs))
{
}

std::string Alignment::to_string() const
{
  std::stringstream ss;
  ss << "Align(" << lhs_->to_string() << ", " << rhs_->to_string() << ")";
  return ss.str();
}

std::shared_ptr<Constraint> align(std::shared_ptr<Variable> lhs, std::shared_ptr<Variable> rhs)
{
  return std::make_shared<Alignment>(std::move(lhs), std::move(rhs));
}

}  // namespace legate
