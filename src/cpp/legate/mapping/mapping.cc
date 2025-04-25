/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/mapping/mapping.h>

#include <legate/mapping/detail/mapping.h>
#include <legate/mapping/detail/store.h>
#include <legate/utilities/detail/traced_exception.h>

#include <stdexcept>
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
  static const DimOrdering ordering{make_internal_shared<detail::DimOrdering>(Kind::C)};
  return ordering;
}

/*static*/ DimOrdering DimOrdering::fortran_order()
{
  static const DimOrdering ordering{make_internal_shared<detail::DimOrdering>(Kind::FORTRAN)};
  return ordering;
}

/*static*/ DimOrdering DimOrdering::custom_order(std::vector<std::int32_t> dims)
{
  return DimOrdering{make_internal_shared<detail::DimOrdering>(std::move(dims))};
}

void DimOrdering::set_c_order() { *this = c_order(); }

void DimOrdering::set_fortran_order() { *this = fortran_order(); }

void DimOrdering::set_custom_order(std::vector<std::int32_t> dims)
{
  impl_ = make_internal_shared<detail::DimOrdering>(std::move(dims));
}

DimOrdering::Kind DimOrdering::kind() const { return impl_->kind; }

std::vector<std::int32_t> DimOrdering::dimensions() const { return impl_->dims; }

bool DimOrdering::operator==(const DimOrdering& other) const { return *impl_ == *other.impl_; }

bool DimOrdering::operator!=(const DimOrdering& other) const { return !(*this == other); }

DimOrdering::~DimOrdering() noexcept = default;

bool InstanceMappingPolicy::operator==(const InstanceMappingPolicy& other) const
{
  return target == other.target && allocation == other.allocation && layout == other.layout &&
         exact == other.exact && redundant == other.redundant && ordering == other.ordering;
}

bool InstanceMappingPolicy::operator!=(const InstanceMappingPolicy& other) const
{
  return !operator==(other);
}

// ==========================================================================================

StoreMapping::StoreMapping(StoreMapping&&) noexcept = default;

StoreMapping& StoreMapping::operator=(StoreMapping&&) noexcept = default;

StoreMapping::~StoreMapping() = default;

StoreMapping::StoreMapping(std::unique_ptr<detail::StoreMapping> impl) : impl_{std::move(impl)} {}

/*static*/ StoreMapping StoreMapping::default_mapping(const Store& store,
                                                      StoreTarget target,
                                                      bool exact)
{
  return create(store, InstanceMappingPolicy{}.with_target(target).with_exact(exact));
}

/*static*/ StoreMapping StoreMapping::create(const Store& store, InstanceMappingPolicy&& policy)
{
  return StoreMapping{detail::StoreMapping::create(store.impl(), std::move(policy))};
}

/*static*/ StoreMapping StoreMapping::create(const std::vector<Store>& stores,
                                             InstanceMappingPolicy&& policy)
{
  if (stores.empty()) {
    throw legate::detail::TracedException<std::invalid_argument>{
      "Invalid to create a store mapping without any store"};
  }
  auto mapping = create(stores.front(), std::move(policy));

  for (std::size_t i = 1; i < stores.size(); ++i) {
    mapping.add_store(stores[i]);
  }
  return mapping;
}

InstanceMappingPolicy& StoreMapping::policy() { return impl_->policy(); }

const InstanceMappingPolicy& StoreMapping::policy() const { return impl()->policy(); }

Store StoreMapping::store() const { return Store{impl()->store()}; }

std::vector<Store> StoreMapping::stores() const
{
  return {impl()->stores().begin(), impl()->stores().end()};
}

void StoreMapping::add_store(const Store& store) { impl_->add_store(store.impl()); }

}  // namespace legate::mapping
