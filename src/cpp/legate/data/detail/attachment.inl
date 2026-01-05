/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/attachment.h>

namespace legate::detail {

inline Attachment::Attachment(Legion::PhysicalRegion physical_region,
                              InternalSharedPtr<ExternalAllocation> allocation)
  : handle_{std::move(physical_region)}, allocations_{{std::move(allocation)}}
{
}

// ==========================================================================================

inline Attachment::Attachment(Legion::ExternalResources external_resources,
                              std::vector<InternalSharedPtr<ExternalAllocation>> allocations)
  : handle_{std::move(external_resources)}, allocations_{std::move(allocations)}
{
}

inline bool Attachment::exists() const noexcept { return !allocations_.empty(); }

}  // namespace legate::detail
