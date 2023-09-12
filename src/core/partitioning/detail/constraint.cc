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

#include "core/partitioning/detail/constraint.h"

#include <sstream>

#include "core/data/scalar.h"
#include "core/operation/detail/operation.h"
#include "core/partitioning/detail/partitioner.h"
#include "core/partitioning/partition.h"
#include "core/utilities/memory.h"

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
  if (lhs_store->unbound() != rhs_store->unbound()) {
    throw std::invalid_argument("Alignment requires the stores to be all normal or all unbound");
  }
  if (lhs_store->unbound()) return;
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

  if (!(is_point_type(func->type(), range->dim()) || is_rect_type(func->type(), range->dim()))) {
    throw std::invalid_argument("Store from which the image partition is derived should have " +
                                std::to_string(range->dim()) + "-D points or rects");
  }
  if (range->transformed()) {
    throw std::runtime_error("Image constraints on transformed stores are not supported yet");
  }
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
  if (src_part->has_launch_domain()) {
    return create_image(src->operation()->find_store(src), src_part);
  } else {
    return create_no_partition();
  }
}

ScaleConstraint::ScaleConstraint(const Shape& factors,
                                 const Variable* var_smaller,
                                 const Variable* var_bigger)
  : factors_(factors), var_smaller_(var_smaller), var_bigger_(var_bigger)
{
}

void ScaleConstraint::find_partition_symbols(std::vector<const Variable*>& partition_symbols) const
{
  partition_symbols.push_back(var_smaller_);
  partition_symbols.push_back(var_bigger_);
}

void ScaleConstraint::validate() const
{
  auto smaller = var_smaller_->operation()->find_store(var_smaller_);
  auto bigger  = var_bigger_->operation()->find_store(var_bigger_);

  if (smaller->dim() != bigger->dim()) {
    throw std::invalid_argument(
      "Scaling constraint requires the stores to have the same number of dimensions");
  }

  if (smaller->dim() != factors_.size()) {
    throw std::invalid_argument(
      "Scaling constraint requires the number of factors to match the number of dimensions");
  }
}

std::string ScaleConstraint::to_string() const
{
  std::stringstream ss;
  ss << "ScaleConstraint(" << factors_ << ", " << var_smaller_->to_string() << ", "
     << var_bigger_->to_string() << ")";
  return std::move(ss).str();
}

std::unique_ptr<Partition> ScaleConstraint::resolve(const detail::Strategy& strategy) const
{
  auto src_part = strategy[var_smaller()];
  return src_part->scale(factors_);
}

BloatConstraint::BloatConstraint(const Variable* var_source,
                                 const Variable* var_bloat,
                                 const Shape& low_offsets,
                                 const Shape& high_offsets)
  : var_source_(var_source),
    var_bloat_(var_bloat),
    low_offsets_(low_offsets),
    high_offsets_(high_offsets)
{
}

void BloatConstraint::find_partition_symbols(std::vector<const Variable*>& partition_symbols) const
{
  partition_symbols.push_back(var_source_);
  partition_symbols.push_back(var_bloat_);
}

void BloatConstraint::validate() const
{
  auto source = var_source_->operation()->find_store(var_source_);
  auto bloat  = var_bloat_->operation()->find_store(var_bloat_);

  if (source->dim() != bloat->dim()) {
    throw std::invalid_argument(
      "Bloating constraint requires the stores to have the same number of dimensions");
  }

  if (source->dim() != low_offsets_.size() || source->dim() != high_offsets_.size()) {
    throw std::invalid_argument(
      "Bloating constraint requires the number of offsets to match the number of dimensions");
  }
}

std::string BloatConstraint::to_string() const
{
  std::stringstream ss;
  ss << "BloatConstraint(" << var_source_->to_string() << ", " << var_bloat_->to_string()
     << ", low: " << low_offsets_ << ", high: " << high_offsets_ << ")";
  return std::move(ss).str();
}

std::unique_ptr<Partition> BloatConstraint::resolve(const detail::Strategy& strategy) const
{
  auto src_part = strategy[var_source()];
  return src_part->bloat(low_offsets_, high_offsets_);
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

std::unique_ptr<ScaleConstraint> scale(const Shape& factors,
                                       const Variable* var_smaller,
                                       const Variable* var_bigger)
{
  return std::make_unique<ScaleConstraint>(factors, var_smaller, var_bigger);
}

std::unique_ptr<BloatConstraint> bloat(const Variable* var_source,
                                       const Variable* var_bloat,
                                       const Shape& low_offsets,
                                       const Shape& high_offsets)
{
  return std::make_unique<BloatConstraint>(var_source, var_bloat, low_offsets, high_offsets);
}

}  // namespace legate::detail

// explicitly instantiate the deleter for std::unique_ptr<detail::Constraint>
namespace legate {
template struct default_delete<detail::Constraint>;
}
