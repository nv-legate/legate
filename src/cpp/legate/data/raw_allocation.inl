/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/raw_allocation.h>

namespace legate {

inline RawAllocation::RawAllocation(InternalSharedPtr<detail::RawAllocation>&& impl)
  : impl_{std::move(impl)}
{
}

inline const SharedPtr<detail::RawAllocation>& RawAllocation::impl() const { return impl_; }

}  // namespace legate
