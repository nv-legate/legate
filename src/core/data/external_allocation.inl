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

#include "core/data/external_allocation.h"

namespace legate {

inline ExternalAllocation::ExternalAllocation(InternalSharedPtr<detail::ExternalAllocation>&& impl)
  : impl_{std::move(impl)}
{
}

inline const SharedPtr<detail::ExternalAllocation>& ExternalAllocation::impl() const
{
  return impl_;
}

/*static*/ inline ExternalAllocation ExternalAllocation::create_sysmem(
  const void* ptr,
  std::size_t size,
  std::optional<ExternalAllocation::Deleter> deleter /*=std::nullopt*/)
{
  return create_sysmem(const_cast<void*>(ptr), size, true, std::move(deleter));
}

/*static*/ inline ExternalAllocation ExternalAllocation::create_zcmem(
  const void* ptr,
  std::size_t size,
  std::optional<ExternalAllocation::Deleter> deleter /*=std::nullopt*/)
{
  return create_zcmem(const_cast<void*>(ptr), size, true, std::move(deleter));
}

/*static*/ inline ExternalAllocation ExternalAllocation::create_fbmem(
  std::uint32_t local_device_id,
  const void* ptr,
  std::size_t size,
  std::optional<ExternalAllocation::Deleter> deleter /*=std::nullopt*/)
{
  return create_fbmem(local_device_id, const_cast<void*>(ptr), size, true, std::move(deleter));
}

}  // namespace legate
