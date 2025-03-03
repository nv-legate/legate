/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/external_allocation.h>
#include <legate/utilities/detail/traced_exception.h>

#include <stdexcept>

namespace legate::detail {

inline ExternalAllocation::ExternalAllocation(
  bool read_only,
  mapping::StoreTarget target,
  void* ptr,
  std::size_t size,
  std::unique_ptr<Realm::ExternalInstanceResource> resource,
  std::optional<Deleter> deleter)
  : read_only_{read_only},
    target_{target},
    ptr_{ptr
           ? ptr
           : throw TracedException<
               std::invalid_argument>{"External allocation cannot be created from a null pointer"}},
    size_{size},
    resource_{std::move(resource)},
    deleter_{std::move(deleter)}
{
}

inline bool ExternalAllocation::read_only() const { return read_only_; }

inline mapping::StoreTarget ExternalAllocation::target() const { return target_; }

inline void* ExternalAllocation::ptr() const { return ptr_; }

inline std::size_t ExternalAllocation::size() const { return size_; }

inline const Realm::ExternalInstanceResource* ExternalAllocation::resource() const
{
  return resource_.get();
}

}  // namespace legate::detail
