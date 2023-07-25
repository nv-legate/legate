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

#include "core/mapping/mapping.h"

#include "core/mapping/detail/mapping.h"
#include "core/mapping/detail/store.h"

namespace legate::mapping {

std::ostream& operator<<(std::ostream& stream, const TaskTarget& target)
{
  switch (target) {
    case TaskTarget::GPU: {
      stream << "GPU";
      break;
    }
    case TaskTarget::OMP: {
      stream << "OMP";
      break;
    }
    case TaskTarget::CPU: {
      stream << "CPU";
      break;
    }
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const StoreTarget& target)
{
  switch (target) {
    case StoreTarget::SYSMEM: {
      stream << "SYSMEM";
      break;
    }
    case StoreTarget::FBMEM: {
      stream << "FBMEM";
      break;
    }
    case StoreTarget::ZCMEM: {
      stream << "ZCMEM";
      break;
    }
    case StoreTarget::SOCKETMEM: {
      stream << "SOCKETMEM";
      break;
    }
  }
  return stream;
}

/*static*/ DimOrdering DimOrdering::c_order()
{
  static DimOrdering ordering(std::make_shared<detail::DimOrdering>(Kind::C));
  return ordering;
}

/*static*/ DimOrdering DimOrdering::fortran_order()
{
  static DimOrdering ordering(std::make_shared<detail::DimOrdering>(Kind::FORTRAN));
  return ordering;
}

/*static*/ DimOrdering DimOrdering::custom_order(const std::vector<int32_t>& dims)
{
  return DimOrdering(std::make_shared<detail::DimOrdering>(dims));
}

void DimOrdering::set_c_order() { *this = c_order(); }

void DimOrdering::set_fortran_order() { *this = fortran_order(); }

void DimOrdering::set_custom_order(const std::vector<int32_t>& dims)
{
  impl_ = std::make_shared<detail::DimOrdering>(dims);
}

DimOrdering::Kind DimOrdering::kind() const { return impl_->kind; }

std::vector<int32_t> DimOrdering::dimensions() const { return impl_->dims; }

bool DimOrdering::operator==(const DimOrdering& other) const { return *impl_ == *other.impl_; }

DimOrdering::DimOrdering() { impl_ = std::move(c_order().impl_); }

DimOrdering::DimOrdering(const DimOrdering&) = default;

DimOrdering& DimOrdering::operator=(const DimOrdering&) = default;

DimOrdering::DimOrdering(DimOrdering&&) = default;

DimOrdering& DimOrdering::operator=(DimOrdering&&) = default;

DimOrdering::DimOrdering(std::shared_ptr<detail::DimOrdering> impl) : impl_(std::move(impl)) {}

const detail::DimOrdering* DimOrdering::impl() const { return impl_.get(); }

InstanceMappingPolicy& InstanceMappingPolicy::with_target(StoreTarget _target) &
{
  set_target(_target);
  return *this;
}

InstanceMappingPolicy&& InstanceMappingPolicy::with_target(StoreTarget _target) const&
{
  return InstanceMappingPolicy(*this).with_target(_target);
}

InstanceMappingPolicy&& InstanceMappingPolicy::with_target(StoreTarget _target) &&
{
  return std::move(this->with_target(_target));
}

InstanceMappingPolicy& InstanceMappingPolicy::with_allocation_policy(AllocPolicy _allocation) &
{
  set_allocation_policy(_allocation);
  return *this;
}

InstanceMappingPolicy&& InstanceMappingPolicy::with_allocation_policy(
  AllocPolicy _allocation) const&
{
  return InstanceMappingPolicy(*this).with_allocation_policy(_allocation);
}

InstanceMappingPolicy&& InstanceMappingPolicy::with_allocation_policy(AllocPolicy _allocation) &&
{
  return std::move(this->with_allocation_policy(_allocation));
}

InstanceMappingPolicy& InstanceMappingPolicy::with_instance_layout(InstLayout _layout) &
{
  set_instance_layout(_layout);
  return *this;
}

InstanceMappingPolicy&& InstanceMappingPolicy::with_instance_layout(InstLayout _layout) const&
{
  return InstanceMappingPolicy(*this).with_instance_layout(_layout);
}

InstanceMappingPolicy&& InstanceMappingPolicy::with_instance_layout(InstLayout _layout) &&
{
  return std::move(this->with_instance_layout(_layout));
}

InstanceMappingPolicy& InstanceMappingPolicy::with_ordering(DimOrdering _ordering) &
{
  set_ordering(_ordering);
  return *this;
}

InstanceMappingPolicy&& InstanceMappingPolicy::with_ordering(DimOrdering _ordering) const&
{
  return InstanceMappingPolicy(*this).with_ordering(_ordering);
}

InstanceMappingPolicy&& InstanceMappingPolicy::with_ordering(DimOrdering _ordering) &&
{
  return std::move(this->with_ordering(_ordering));
}

InstanceMappingPolicy& InstanceMappingPolicy::with_exact(bool _exact) &
{
  set_exact(_exact);
  return *this;
}

InstanceMappingPolicy&& InstanceMappingPolicy::with_exact(bool _exact) const&
{
  return InstanceMappingPolicy(*this).with_exact(_exact);
}

InstanceMappingPolicy&& InstanceMappingPolicy::with_exact(bool _exact) &&
{
  return std::move(this->with_exact(_exact));
}

void InstanceMappingPolicy::set_target(StoreTarget _target) { target = _target; }

void InstanceMappingPolicy::set_allocation_policy(AllocPolicy _allocation)
{
  allocation = _allocation;
}

void InstanceMappingPolicy::set_instance_layout(InstLayout _layout) { layout = _layout; }

void InstanceMappingPolicy::set_ordering(DimOrdering _ordering) { ordering = _ordering; }

void InstanceMappingPolicy::set_exact(bool _exact) { exact = _exact; }

bool InstanceMappingPolicy::subsumes(const InstanceMappingPolicy& other) const
{
  // the allocation policy doesn't concern the instance layout
  return target == other.target && layout == other.layout && ordering == other.ordering &&
         (exact || !other.exact);
}

InstanceMappingPolicy::InstanceMappingPolicy() {}

InstanceMappingPolicy::~InstanceMappingPolicy() {}

InstanceMappingPolicy::InstanceMappingPolicy(const InstanceMappingPolicy&) = default;

InstanceMappingPolicy& InstanceMappingPolicy::operator=(const InstanceMappingPolicy&) = default;

InstanceMappingPolicy::InstanceMappingPolicy(InstanceMappingPolicy&&) = default;

InstanceMappingPolicy& InstanceMappingPolicy::operator=(InstanceMappingPolicy&&) = default;

bool InstanceMappingPolicy::operator==(const InstanceMappingPolicy& other) const
{
  return target == other.target && allocation == other.allocation && layout == other.layout &&
         exact == other.exact && ordering == other.ordering;
}

bool InstanceMappingPolicy::operator!=(const InstanceMappingPolicy& other) const
{
  return !operator==(other);
}

/*static*/ StoreMapping StoreMapping::default_mapping(Store store, StoreTarget target, bool exact)
{
  return create(store, InstanceMappingPolicy{}.with_target(target).with_exact(exact));
}

/*static*/ StoreMapping StoreMapping::create(Store store, InstanceMappingPolicy&& policy)
{
  return StoreMapping(detail::StoreMapping::create(store.impl(), std::move(policy)).release());
}

/*static*/ StoreMapping StoreMapping::create(const std::vector<Store>& stores,
                                             InstanceMappingPolicy&& policy)
{
  if (stores.empty()) {
    throw std::invalid_argument("Invalid to create a store mapping without any store");
  }
  auto mapping    = new detail::StoreMapping();
  mapping->policy = std::move(policy);
  for (auto& store : stores) { mapping->stores.push_back(store.impl()); }
  return StoreMapping(mapping);
}

InstanceMappingPolicy& StoreMapping::policy() { return impl_->policy; }

const InstanceMappingPolicy& StoreMapping::policy() const { return impl_->policy; }

Store StoreMapping::store() const { return Store(impl_->stores.front()); }

std::vector<Store> StoreMapping::stores() const
{
  std::vector<Store> result;
  for (auto& store : impl_->stores) { result.emplace_back(store); }
  return std::move(result);
}

void StoreMapping::add_store(Store store) { impl_->stores.push_back(store.impl()); }

StoreMapping::StoreMapping(detail::StoreMapping* impl) : impl_(impl) {}

const detail::StoreMapping* StoreMapping::impl() const { return impl_; }

StoreMapping::StoreMapping(StoreMapping&& other) : impl_(other.impl_) { other.impl_ = nullptr; }

StoreMapping& StoreMapping::operator=(StoreMapping&& other)
{
  impl_       = other.impl_;
  other.impl_ = nullptr;
  return *this;
}

StoreMapping::~StoreMapping() { delete impl_; }

detail::StoreMapping* StoreMapping::release()
{
  auto result = impl_;
  impl_       = nullptr;
  return result;
}

}  // namespace legate::mapping
