/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/external_allocation.h>

namespace legate {

inline ExternalAllocation::ExternalAllocation(InternalSharedPtr<detail::ExternalAllocation>&& impl)
  : impl_{std::move(impl)}
{
}

inline const SharedPtr<detail::ExternalAllocation>& ExternalAllocation::impl() const
{
  return impl_;
}

}  // namespace legate
