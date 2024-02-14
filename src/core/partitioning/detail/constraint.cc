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

#include "core/operation/detail/operation.h"
#include "core/partitioning/detail/partitioner.h"
#include "core/partitioning/partition.h"
#include "core/utilities/memory.h"

#include <sstream>

namespace legate::detail {

Literal::Literal(InternalSharedPtr<Partition> partition) : partition_{std::move(partition)} {}

std::string Literal::to_string() const { return partition_->to_string(); }

void Literal::find_partition_symbols(std::vector<const Variable*>& /*partition_symbols*/) const {}

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

void Alignment::find_partition_symbols(std::vector<const Variable*>& partition_symbols) const
{
  partition_symbols.push_back(lhs_);
  partition_symbols.push_back(rhs_);
}

void Alignment::validate() const
{
  if (is_trivial()) {
    return;
  }

  auto lhs_store = lhs_->operation()->find_store(lhs_);
  auto rhs_store = rhs_->operation()->find_store(rhs_);
  if (lhs_store->unbound() != rhs_store->unbound()) {
    throw std::invalid_argument{"Alignment requires the stores to be all normal or all unbound"};
  }
  if (lhs_store->unbound()) {
    return;
  }
  if (*lhs_store->shape() != *rhs_store->shape()) {
    throw std::invalid_argument{"Alignment requires the stores to have the same shape, but found " +
                                lhs_store->extents().to_string() + " and " +
                                rhs_store->extents().to_string()};
  }
}

std::string Alignment::to_string() const
{
  std::stringstream ss;

  ss << "Align(" << lhs_->to_string() << ", " << rhs_->to_string() << ")";
  return std::move(ss).str();
}

void Broadcast::find_partition_symbols(std::vector<const Variable*>& partition_symbols) const
{
  partition_symbols.push_back(variable_);
}

void Broadcast::validate() const
{
  if (axes_.empty()) {
    return;
  }
  auto store = variable_->operation()->find_store(variable_);
  for (auto axis : axes_.data()) {
    if (axis >= store->dim()) {
      throw std::invalid_argument{"Invalid broadcasting dimension " + std::to_string(axis) +
                                  " for a " + std::to_string(store->dim()) + "-D store"};
    }
  }
}

std::string Broadcast::to_string() const
{
  std::stringstream ss;

  if (axes_.empty()) {
    ss << "Broadcast(" << variable_->to_string() << ")";
  } else {
    ss << "Broadcast(" << variable_->to_string() << ", " << axes_.to_string() << ")";
  }
  return std::move(ss).str();
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
    throw std::invalid_argument{"Store from which the image partition is derived should have " +
                                std::to_string(range->dim()) + "-D points or rects"};
  }
  if (range->transformed()) {
    throw std::runtime_error{"Image constraints on transformed stores are not supported yet"};
  }
}

std::string ImageConstraint::to_string() const
{
  std::stringstream ss;

  ss << "ImageConstraint(" << var_function_->to_string() << ", " << var_range_->to_string() << ")";
  return std::move(ss).str();
}

InternalSharedPtr<Partition> ImageConstraint::resolve(const detail::Strategy& strategy) const
{
  const auto* src = var_function();
  auto src_part   = strategy[src];
  if (src_part->has_launch_domain()) {
    auto* op = src->operation();
    return create_image(op->find_store(src).as_user_ptr(), src_part.as_user_ptr(), op->machine());
  }
  return create_no_partition();
}

void ScaleConstraint::find_partition_symbols(std::vector<const Variable*>& partition_symbols) const
{
  partition_symbols.push_back(var_smaller_);
  partition_symbols.push_back(var_bigger_);
}

void ScaleConstraint::validate() const
{
  auto smaller    = var_smaller_->operation()->find_store(var_smaller_);
  auto bigger     = var_bigger_->operation()->find_store(var_bigger_);
  const auto sdim = smaller->dim();

  if (sdim != bigger->dim()) {
    throw std::invalid_argument{

      "Scaling constraint requires the stores to have the same number of dimensions"};
  }

  if (const auto sdim_s = static_cast<std::size_t>(sdim); sdim_s != factors_.size()) {
    throw std::invalid_argument{
      "Scaling constraint requires the number of factors to match the number of dimensions"};
  }
}

std::string ScaleConstraint::to_string() const
{
  std::stringstream ss;

  ss << "ScaleConstraint(" << factors_ << ", " << var_smaller_->to_string() << ", "
     << var_bigger_->to_string() << ")";
  return std::move(ss).str();
}

InternalSharedPtr<Partition> ScaleConstraint::resolve(const detail::Strategy& strategy) const
{
  return strategy[var_smaller()]->scale(factors_);
}

void BloatConstraint::find_partition_symbols(std::vector<const Variable*>& partition_symbols) const
{
  partition_symbols.push_back(var_source_);
  partition_symbols.push_back(var_bloat_);
}

void BloatConstraint::validate() const
{
  auto source     = var_source_->operation()->find_store(var_source_);
  auto bloat      = var_bloat_->operation()->find_store(var_bloat_);
  const auto sdim = source->dim();

  if (sdim != bloat->dim()) {
    throw std::invalid_argument{
      "Bloating constraint requires the stores to have the same number of dimensions"};
  }

  if (const auto sdim_s = static_cast<std::size_t>(sdim);
      sdim_s != low_offsets_.size() || sdim_s != high_offsets_.size()) {
    throw std::invalid_argument{
      "Bloating constraint requires the number of offsets to match the number of dimensions"};
  }
}

std::string BloatConstraint::to_string() const
{
  std::stringstream ss;

  ss << "BloatConstraint(" << var_source_->to_string() << ", " << var_bloat_->to_string()
     << ", low: " << low_offsets_ << ", high: " << high_offsets_ << ")";
  return std::move(ss).str();
}

InternalSharedPtr<Partition> BloatConstraint::resolve(const detail::Strategy& strategy) const
{
  return strategy[var_source()]->bloat(low_offsets_, high_offsets_);
}

InternalSharedPtr<Alignment> align(const Variable* lhs, const Variable* rhs)
{
  return make_internal_shared<Alignment>(lhs, rhs);
}

[[nodiscard]] InternalSharedPtr<Broadcast> broadcast(const Variable* variable)
{
  return make_internal_shared<Broadcast>(variable);
}

InternalSharedPtr<Broadcast> broadcast(const Variable* variable, tuple<std::uint32_t> axes)
{
  if (axes.empty()) {
    throw std::invalid_argument{"List of axes to broadcast must not be empty"};
  }
  return make_internal_shared<Broadcast>(variable, std::move(axes));
}

InternalSharedPtr<ImageConstraint> image(const Variable* var_function, const Variable* var_range)
{
  return make_internal_shared<ImageConstraint>(var_function, var_range);
}

InternalSharedPtr<ScaleConstraint> scale(tuple<std::uint64_t> factors,
                                         const Variable* var_smaller,
                                         const Variable* var_bigger)
{
  return make_internal_shared<ScaleConstraint>(std::move(factors), var_smaller, var_bigger);
}

InternalSharedPtr<BloatConstraint> bloat(const Variable* var_source,
                                         const Variable* var_bloat,
                                         tuple<std::uint64_t> low_offsets,
                                         tuple<std::uint64_t> high_offsets)
{
  return make_internal_shared<BloatConstraint>(
    var_source, var_bloat, std::move(low_offsets), std::move(high_offsets));
}

}  // namespace legate::detail
