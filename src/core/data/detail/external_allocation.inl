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

#include "core/data/detail/external_allocation.h"

namespace legate::detail {

inline ExternalAllocation::ExternalAllocation(
  bool read_only,
  mapping::StoreTarget target,
  void* ptr,
  size_t size,
  std::unique_ptr<Realm::ExternalInstanceResource> resource,
  std::optional<Deleter> deleter)
  : read_only_{read_only},
    target_{target},
    ptr_{
      ptr
        ? ptr
        : throw std::invalid_argument{"External allocation cannot be created from a null pointer"}},
    size_{size},
    resource_{std::move(resource)},
    deleter_{std::move(deleter)}
{
}

inline bool ExternalAllocation::read_only() const { return read_only_; }

inline mapping::StoreTarget ExternalAllocation::target() const { return target_; }

inline void* ExternalAllocation::ptr() const { return ptr_; }

inline size_t ExternalAllocation::size() const { return size_; }

inline const Realm::ExternalInstanceResource* ExternalAllocation::resource() const
{
  return resource_.get();
}

}  // namespace legate::detail
