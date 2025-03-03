/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/mapping/mapping.h>

namespace legate::mapping {

inline const SharedPtr<detail::DimOrdering>& DimOrdering::impl() const noexcept { return impl_; }

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

inline InstanceMappingPolicy& InstanceMappingPolicy::with_redundant(bool _redundant) &
{
  set_redundant(_redundant);
  return *this;
}

inline InstanceMappingPolicy InstanceMappingPolicy::with_redundant(bool _redundant) const&
{
  return InstanceMappingPolicy{*this}.with_redundant(_redundant);
}

inline InstanceMappingPolicy&& InstanceMappingPolicy::with_redundant(bool _redundant) &&
{
  return std::move(with_redundant(_redundant));
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

inline void InstanceMappingPolicy::set_redundant(bool _redundant) { redundant = _redundant; }

// ==========================================================================================

inline const detail::StoreMapping* StoreMapping::impl() const noexcept { return impl_.get(); }

inline detail::StoreMapping* StoreMapping::release_(ReleaseKey) noexcept { return impl_.release(); }

}  // namespace legate::mapping
