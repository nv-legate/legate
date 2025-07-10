/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/mapping/detail/store.h>

#include <legate/data/detail/transform.h>

#include <utility>

namespace legate::mapping::detail {

bool RegionField::can_colocate_with(const RegionField& other) const
{
  return get_requirement().region.get_tree_id() == other.get_requirement().region.get_tree_id() &&
         field_id() == other.field_id() && dim() == other.dim();
}

Domain RegionField::domain(Legion::Mapping::MapperRuntime& runtime,
                           Legion::Mapping::MapperContext context) const
{
  return runtime.get_index_space_domain(context, get_index_space());
}

Legion::IndexSpace RegionField::get_index_space() const
{
  return get_requirement().region.get_index_space();
}

Store::Store(std::int32_t dim,
             InternalSharedPtr<legate::detail::Type> type,
             FutureWrapper future,
             InternalSharedPtr<legate::detail::TransformStack>&& transform)
  : is_future_{true},
    dim_{dim},
    type_{std::move(type)},
    future_{std::move(future)},
    transform_{std::move(transform)}
{
}

Store::Store(Legion::Mapping::MapperRuntime& runtime,
             Legion::Mapping::MapperContext context,
             std::int32_t dim,
             InternalSharedPtr<legate::detail::Type> type,
             GlobalRedopID redop_id,
             const RegionField& region_field,
             bool is_unbound_store,
             InternalSharedPtr<legate::detail::TransformStack>&& transform)
  : is_unbound_store_{is_unbound_store},
    dim_{dim},
    type_{std::move(type)},
    redop_id_{redop_id},
    region_field_{region_field},
    transform_{std::move(transform)},
    runtime_{&runtime},
    context_{context}
{
}

Store::Store(Legion::Mapping::MapperRuntime& runtime,
             Legion::Mapping::MapperContext context,
             const Legion::RegionRequirement& requirement)
  : dim_{requirement.region.get_dim()},
    region_field_{requirement, dim_, 0, requirement.instance_fields.front(), false /*unbound*/},
    runtime_{&runtime},
    context_{context}
{
}

bool Store::valid() const { return is_future() || unbound() || region_field().valid(); }

bool Store::can_colocate_with(const Store& other) const
{
  if (is_future() || other.is_future()) {
    return false;
  }
  if (unbound() || other.unbound()) {
    return false;
  }
  if (is_reduction() || other.is_reduction()) {
    return redop() == other.redop() && region_field().can_colocate_with(other.region_field());
  }
  return region_field().can_colocate_with(other.region_field());
}

const RegionField& Store::region_field() const
{
  LEGATE_ASSERT(!is_future());
  return region_field_;
}

const FutureWrapper& Store::future() const
{
  LEGATE_ASSERT(is_future());
  return future_;
}

RegionField::Id Store::unique_region_field_id() const { return region_field().unique_id(); }

std::uint32_t Store::requirement_index() const { return region_field().index(); }

std::uint32_t Store::future_index() const { return future().index(); }

Domain Store::domain() const
{
  LEGATE_CHECK(!unbound());

  auto result = [&] {
    if (is_future()) {
      return future().domain();
    }
    // NOLINTNEXTLINE(bugprone-unchecked-optional-value)
    return region_field().domain(*runtime_, context_);
  }();

  if (!transform_->identity()) {
    result = transform_->transform(result);
  }
  LEGATE_CHECK(result.dim == dim());
  return result;
}

std::vector<std::int32_t> Store::find_imaginary_dims() const
{
  if (transform_) {
    return transform_->find_imaginary_dims();
  }
  return {};
}

tuple<std::int32_t> Store::invert_dims(tuple<std::int32_t> dims) const
{
  if (transform_) {
    return transform_->invert_dims(std::move(dims));
  }
  return dims;
}

}  // namespace legate::mapping::detail
