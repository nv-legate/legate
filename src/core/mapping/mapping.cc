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

#include <utility>

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
  static const DimOrdering ordering{std::make_shared<detail::DimOrdering>(Kind::C)};
  return ordering;
}

/*static*/ DimOrdering DimOrdering::fortran_order()
{
  static const DimOrdering ordering{std::make_shared<detail::DimOrdering>(Kind::FORTRAN)};
  return ordering;
}

/*static*/ DimOrdering DimOrdering::custom_order(std::vector<int32_t> dims)
{
  return DimOrdering{std::make_shared<detail::DimOrdering>(std::move(dims))};
}

void DimOrdering::set_c_order() { *this = c_order(); }

void DimOrdering::set_fortran_order() { *this = fortran_order(); }

void DimOrdering::set_custom_order(std::vector<int32_t> dims)
{
  impl_ = std::make_shared<detail::DimOrdering>(std::move(dims));
}

DimOrdering::Kind DimOrdering::kind() const { return impl_->kind; }

std::vector<int32_t> DimOrdering::dimensions() const { return impl_->dims; }

bool DimOrdering::operator==(const DimOrdering& other) const { return *impl_ == *other.impl_; }

bool InstanceMappingPolicy::subsumes(const InstanceMappingPolicy& other) const
{
  // the allocation policy doesn't concern the instance layout
  return target == other.target && layout == other.layout && ordering == other.ordering &&
         (exact || !other.exact);
}

bool InstanceMappingPolicy::operator==(const InstanceMappingPolicy& other) const
{
  return target == other.target && allocation == other.allocation && layout == other.layout &&
         exact == other.exact && ordering == other.ordering;
}

bool InstanceMappingPolicy::operator!=(const InstanceMappingPolicy& other) const
{
  return !operator==(other);
}

/*static*/ StoreMapping StoreMapping::default_mapping(const Store& store,
                                                      StoreTarget target,
                                                      bool exact)
{
  return create(store, InstanceMappingPolicy{}.with_target(target).with_exact(exact));
}

/*static*/ StoreMapping StoreMapping::create(const Store& store, InstanceMappingPolicy&& policy)
{
  return StoreMapping{detail::StoreMapping::create(store.impl(), std::move(policy)).release()};
}

/*static*/ StoreMapping StoreMapping::create(const std::vector<Store>& stores,
                                             InstanceMappingPolicy&& policy)
{
  if (stores.empty()) {
    throw std::invalid_argument{"Invalid to create a store mapping without any store"};
  }
  auto mapping = std::make_unique<detail::StoreMapping>();

  mapping->policy = std::move(policy);
  mapping->stores.reserve(stores.size());
  for (auto&& store : stores) {
    mapping->stores.emplace_back(store.impl());
  }
  return StoreMapping{mapping.release()};
}

InstanceMappingPolicy& StoreMapping::policy() { return impl_->policy; }

const InstanceMappingPolicy& StoreMapping::policy() const { return impl_->policy; }

Store StoreMapping::store() const { return Store{impl_->stores.front()}; }

std::vector<Store> StoreMapping::stores() const
{
  return {impl_->stores.begin(), impl_->stores.end()};
}

void StoreMapping::add_store(const Store& store) { impl_->stores.push_back(store.impl()); }

StoreMapping::StoreMapping(detail::StoreMapping* impl) noexcept : impl_{impl} {}

void StoreMapping::StoreMappingImplDeleter::operator()(detail::StoreMapping* ptr) const noexcept
{
  delete ptr;
}

}  // namespace legate::mapping
