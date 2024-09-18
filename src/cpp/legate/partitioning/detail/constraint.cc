/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "legate/partitioning/detail/constraint.h"

#include "legate/operation/detail/operation.h"
#include "legate/partitioning/detail/partition.h"
#include "legate/partitioning/detail/partitioner.h"
#include "legate/utilities/memory.h"

#include <fmt/format.h>
#include <fmt/ranges.h>

namespace legate::detail {

void Variable::find_partition_symbols(std::vector<const Variable*>& partition_symbols) const
{
  partition_symbols.push_back(this);
}

std::string Variable::to_string() const { return fmt::format("X{}{{{}}}", id(), *operation()); }

const InternalSharedPtr<LogicalStore>& Variable::store() const { return op_->find_store(this); }

void Alignment::find_partition_symbols(std::vector<const Variable*>& partition_symbols) const
{
  partition_symbols.reserve(partition_symbols.size() + 2);
  partition_symbols.push_back(lhs_);
  partition_symbols.push_back(rhs_);
}

void Alignment::validate() const
{
  if (is_trivial()) {
    return;
  }

  auto&& lhs_store = lhs_->operation()->find_store(lhs_);
  auto&& rhs_store = rhs_->operation()->find_store(rhs_);
  if (lhs_store->unbound() != rhs_store->unbound()) {
    throw std::invalid_argument{"Alignment requires the stores to be all normal or all unbound"};
  }
  if (lhs_store->unbound()) {
    return;
  }
  if (*lhs_store->shape() != *rhs_store->shape()) {
    throw std::invalid_argument{
      fmt::format("Alignment requires the stores to have the same shape, but found {} and {}",
                  lhs_store->extents(),
                  rhs_store->extents())};
  }
}

std::string Alignment::to_string() const { return fmt::format("Align({}, {})", *lhs(), *rhs()); }

void Broadcast::find_partition_symbols(std::vector<const Variable*>& partition_symbols) const
{
  partition_symbols.push_back(variable_);
}

void Broadcast::validate() const
{
  if (axes_.empty()) {
    return;
  }
  auto&& store = variable_->operation()->find_store(variable_);
  for (auto axis : axes_.data()) {
    if (axis >= store->dim()) {
      throw std::invalid_argument{
        fmt::format("Invalid broadcasting dimension {} for a {}-D store", axis, store->dim())};
    }
  }
}

std::string Broadcast::to_string() const
{
  if (axes_.empty()) {
    return fmt::format("Broadcast({})", *variable());
  }
  return fmt::format("Broadcast({}, {})", *variable(), axes());
}

void ImageConstraint::find_partition_symbols(std::vector<const Variable*>& partition_symbols) const
{
  partition_symbols.push_back(var_function_);
  partition_symbols.push_back(var_range_);
}

void ImageConstraint::validate() const
{
  auto&& func  = var_function_->operation()->find_store(var_function_);
  auto&& range = var_range_->operation()->find_store(var_range_);

  if (!(is_point_type(func->type(), range->dim()) || is_rect_type(func->type(), range->dim()))) {
    throw std::invalid_argument{fmt::format(
      "Store from which the image partition is derived should have {}-D points or rects",
      range->dim())};
  }
  if (range->transformed()) {
    throw std::runtime_error{"Image constraints on transformed stores are not supported yet"};
  }
}

std::string ImageConstraint::to_string() const
{
  return fmt::format("ImageConstraint({}, {})", *var_function(), *var_range());
}

InternalSharedPtr<Partition> ImageConstraint::resolve(const detail::Strategy& strategy) const
{
  const auto* src = var_function();
  auto&& src_part = strategy[src];
  if (src_part->has_launch_domain()) {
    auto* op = src->operation();
    return create_image(
      op->find_store(src).as_user_ptr(), src_part.as_user_ptr(), op->machine(), hint_);
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
  auto&& smaller  = var_smaller_->operation()->find_store(var_smaller_);
  auto&& bigger   = var_bigger_->operation()->find_store(var_bigger_);
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
  return fmt::format("ScaleConstraint({}, {}, {})", factors(), *var_smaller(), *var_bigger());
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
  auto&& source   = var_source_->operation()->find_store(var_source_);
  auto&& bloat    = var_bloat_->operation()->find_store(var_bloat_);
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
  return fmt::format("BloatConstraint({}, {}, low: {}, high: {})",
                     *var_source(),
                     *var_bloat(),
                     low_offsets(),
                     high_offsets());
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

InternalSharedPtr<ImageConstraint> image(const Variable* var_function,
                                         const Variable* var_range,
                                         ImageComputationHint hint)
{
  return make_internal_shared<ImageConstraint>(var_function, var_range, hint);
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
