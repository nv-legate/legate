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

#include "core/partitioning/detail/constraint.h"

#include <sstream>

#include "core/data/scalar.h"
#include "core/operation/detail/operation.h"
#include "core/partitioning/detail/partitioner.h"
#include "core/partitioning/partition.h"

namespace legate::detail {

Literal::Literal(const std::shared_ptr<Partition>& partition) : partition_(partition) {}

std::string Literal::to_string() const { return partition_->to_string(); }

void Literal::find_partition_symbols(std::vector<const Variable*>& partition_symbols) const {}

Variable::Variable(const detail::Operation* op, int32_t id) : op_(op), id_(id) {}

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

Alignment::Alignment(const Variable* lhs, const Variable* rhs) : lhs_(lhs), rhs_(rhs) {}

void Alignment::find_partition_symbols(std::vector<const Variable*>& partition_symbols) const
{
  partition_symbols.push_back(lhs_);
  partition_symbols.push_back(rhs_);
}

void Alignment::validate() const
{
  if (*lhs_ == *rhs_) return;
  auto lhs_store = lhs_->operation()->find_store(lhs_);
  auto rhs_store = rhs_->operation()->find_store(rhs_);
  if (lhs_store->extents() != rhs_store->extents())
    throw std::invalid_argument("Alignment requires the stores to have the same shape, but found " +
                                lhs_store->extents().to_string() + " and " +
                                rhs_store->extents().to_string());
}

std::string Alignment::to_string() const
{
  std::stringstream ss;
  ss << "Align(" << lhs_->to_string() << ", " << rhs_->to_string() << ")";
  return std::move(ss).str();
}

Broadcast::Broadcast(const Variable* variable, const tuple<int32_t>& axes)
  : variable_(variable), axes_(axes)
{
}

void Broadcast::find_partition_symbols(std::vector<const Variable*>& partition_symbols) const
{
  partition_symbols.push_back(variable_);
}

void Broadcast::validate() const
{
  auto store = variable_->operation()->find_store(variable_);
  for (auto axis : axes_.data()) {
    if (axis < 0 || axis >= store->dim())
      throw std::invalid_argument("Invalid broadcasting dimension " + std::to_string(axis) +
                                  " for a " + std::to_string(store->dim()) + "-D store");
  }
}

std::string Broadcast::to_string() const
{
  std::stringstream ss;
  ss << "Broadcast(" << variable_->to_string() << ", " << axes_.to_string() << ")";
  return std::move(ss).str();
}

ImageConstraint::ImageConstraint(const Variable* var_function, const Variable* var_range)
  : var_function_(var_function), var_range_(var_range)
{
}

void ImageConstraint::find_partition_symbols(std::vector<const Variable*>& partition_symbols) const
{
  partition_symbols.push_back(var_function_);
  partition_symbols.push_back(var_range_);
}

void ImageConstraint::validate() const
{
  auto func  = var_function_->operation()->find_store(var_function_);
  auto range = var_range_->operation()->find_store(var_range_);

  if (!(is_point_type(func->type(), range->dim()) || is_rect_type(func->type(), range->dim())))
    throw std::invalid_argument("Store from which the image partition is derived should have " +
                                std::to_string(range->dim()) + "-D points or rects");
}

std::string ImageConstraint::to_string() const
{
  std::stringstream ss;
  ss << "ImageConstraint(" << var_function_->to_string() << ", " << var_range_->to_string() << ")";
  return std::move(ss).str();
}

std::unique_ptr<Partition> ImageConstraint::resolve(const detail::Strategy& strategy) const
{
  const auto* src = var_function();
  auto src_part   = strategy[src];
  if (src_part->has_launch_domain())
    return create_image(src->operation()->find_store(src), src_part);
  else
    return create_no_partition();
}

std::unique_ptr<Alignment> align(const Variable* lhs, const Variable* rhs)
{
  return std::make_unique<Alignment>(lhs, rhs);
}

std::unique_ptr<Broadcast> broadcast(const Variable* variable, const tuple<int32_t>& axes)

{
  return std::make_unique<Broadcast>(variable, axes);
}

std::unique_ptr<ImageConstraint> image(const Variable* var_function, const Variable* var_range)
{
  return std::make_unique<ImageConstraint>(var_function, var_range);
}

}  // namespace legate::detail
