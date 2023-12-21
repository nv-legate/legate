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

#pragma once

#include "core/mapping/mapping.h"

namespace legate::mapping {

inline const detail::DimOrdering* DimOrdering::impl() const noexcept { return impl_.get(); }

inline DimOrdering::DimOrdering(InternalSharedPtr<detail::DimOrdering> impl)
  : impl_{std::move(impl)}
{
}

// ==========================================================================================

inline InstanceMappingPolicy& InstanceMappingPolicy::with_target(StoreTarget _target) &
{
  set_target(_target);
  return *this;
}

inline InstanceMappingPolicy InstanceMappingPolicy::with_target(StoreTarget _target) const&
{
  return InstanceMappingPolicy{*this}.with_target(_target);
}

inline InstanceMappingPolicy&& InstanceMappingPolicy::with_target(StoreTarget _target) &&
{
  return std::move(with_target(_target));
}

inline InstanceMappingPolicy& InstanceMappingPolicy::with_allocation_policy(
  AllocPolicy _allocation) &
{
  set_allocation_policy(_allocation);
  return *this;
}

inline InstanceMappingPolicy InstanceMappingPolicy::with_allocation_policy(
  AllocPolicy _allocation) const&
{
  return InstanceMappingPolicy{*this}.with_allocation_policy(_allocation);
}

inline InstanceMappingPolicy&& InstanceMappingPolicy::with_allocation_policy(
  AllocPolicy _allocation) &&
{
  return std::move(with_allocation_policy(_allocation));
}

inline InstanceMappingPolicy& InstanceMappingPolicy::with_instance_layout(InstLayout _layout) &
{
  set_instance_layout(_layout);
  return *this;
}

inline InstanceMappingPolicy InstanceMappingPolicy::with_instance_layout(InstLayout _layout) const&
{
  return InstanceMappingPolicy{*this}.with_instance_layout(_layout);
}

inline InstanceMappingPolicy&& InstanceMappingPolicy::with_instance_layout(InstLayout _layout) &&
{
  return std::move(with_instance_layout(_layout));
}

inline InstanceMappingPolicy& InstanceMappingPolicy::with_ordering(DimOrdering _ordering) &
{
  set_ordering(std::move(_ordering));
  return *this;
}

inline InstanceMappingPolicy InstanceMappingPolicy::with_ordering(DimOrdering _ordering) const&
{
  return InstanceMappingPolicy{*this}.with_ordering(std::move(_ordering));
}

inline InstanceMappingPolicy&& InstanceMappingPolicy::with_ordering(DimOrdering _ordering) &&
{
  return std::move(with_ordering(std::move(_ordering)));
}

inline InstanceMappingPolicy& InstanceMappingPolicy::with_exact(bool _exact) &
{
  set_exact(_exact);
  return *this;
}

inline InstanceMappingPolicy InstanceMappingPolicy::with_exact(bool _exact) const&
{
  return InstanceMappingPolicy{*this}.with_exact(_exact);
}

inline InstanceMappingPolicy&& InstanceMappingPolicy::with_exact(bool _exact) &&
{
  return std::move(with_exact(_exact));
}

inline void InstanceMappingPolicy::set_target(StoreTarget _target) { target = _target; }

inline void InstanceMappingPolicy::set_allocation_policy(AllocPolicy _allocation)
{
  allocation = _allocation;
}

inline void InstanceMappingPolicy::set_instance_layout(InstLayout _layout) { layout = _layout; }

inline void InstanceMappingPolicy::set_ordering(DimOrdering _ordering)
{
  ordering = std::move(_ordering);
}

inline void InstanceMappingPolicy::set_exact(bool _exact) { exact = _exact; }

// ==========================================================================================

inline const detail::StoreMapping* StoreMapping::impl() const noexcept { return impl_.get(); }

inline detail::StoreMapping* StoreMapping::release() noexcept { return impl_.release(); }

}  // namespace legate::mapping
