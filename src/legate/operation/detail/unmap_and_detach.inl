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

#include "legate/operation/detail/unmap_and_detach.h"

namespace legate::detail {

inline UnmapAndDetach::UnmapAndDetach(
  std::uint64_t unique_id,
  InternalSharedPtr<LogicalRegionField::PhysicalState> physical_state,
  bool unordered)
  : Operation{unique_id}, physical_state_{std::move(physical_state)}, unordered_{unordered}
{
}

inline Operation::Kind UnmapAndDetach::kind() const { return Kind::UNMAP_AND_DETACH; }

}  // namespace legate::detail
