/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "legate/data/detail/attachment.h"

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
