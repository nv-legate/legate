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

#include "core/data/detail/logical_region_field.h"

#include <utility>

namespace legate::detail {

inline LogicalRegionField::LogicalRegionField(FieldManager* manager,
                                              const Legion::LogicalRegion& lr,
                                              Legion::FieldID fid,
                                              InternalSharedPtr<LogicalRegionField> parent)
  : manager_{manager}, lr_{lr}, fid_{fid}, parent_{std::move(parent)}
{
}

inline const Legion::LogicalRegion& LogicalRegionField::region() const { return lr_; }

inline Legion::FieldID LogicalRegionField::field_id() const { return fid_; }

template <typename T>
void LogicalRegionField::add_invalidation_callback(T&& callback)
{
  static_assert(std::is_nothrow_invocable_v<T>);
  add_invalidation_callback_(std::forward<T>(callback));
}

}  // namespace legate::detail
