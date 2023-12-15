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

#include "core/data/detail/attachment.h"

namespace legate::detail {

inline SingleAttachment::SingleAttachment(Legion::PhysicalRegion* physical_region,
                                          InternalSharedPtr<ExternalAllocation> allocation)
  : physical_region_{physical_region}, allocation_{std::move(allocation)}
{
}

inline SingleAttachment::~SingleAttachment() { maybe_deallocate(); }

// ==========================================================================================

inline IndexAttachment::IndexAttachment(
  const Legion::ExternalResources& external_resources,
  std::vector<InternalSharedPtr<ExternalAllocation>> allocations)
  : external_resources_{std::make_unique<Legion::ExternalResources>(external_resources)},
    allocations_{std::move(allocations)}
{
}

}  // namespace legate::detail
