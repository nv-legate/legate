/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/logical_region_field.h>

#include <utility>

namespace legate::detail {

inline void LogicalRegionField::PhysicalState::set_physical_region(
  Legion::PhysicalRegion physical_region)
{
  physical_region_ = std::move(physical_region);
}

inline void LogicalRegionField::PhysicalState::set_attachment(Attachment attachment)
{
  attachment_ = std::move(attachment);
}

inline void LogicalRegionField::PhysicalState::set_has_pending_detach(bool has_pending_detach)
{
  has_pending_detach_ = has_pending_detach;
}

inline void LogicalRegionField::PhysicalState::add_callback(std::function<void()> callback)
{
  callbacks_.push_back(std::move(callback));
}

inline bool LogicalRegionField::PhysicalState::has_attachment() const
{
  return attachment().exists();
}

inline const Legion::PhysicalRegion& LogicalRegionField::PhysicalState::physical_region() const
{
  return physical_region_;
}

inline const Attachment& LogicalRegionField::PhysicalState::attachment() const
{
  return attachment_;
}

// ==========================================================================================

inline LogicalRegionField::LogicalRegionField(
  InternalSharedPtr<Shape> shape,
  std::uint32_t field_size,
  Legion::LogicalRegion lr,
  Legion::FieldID fid,
  std::optional<InternalSharedPtr<LogicalRegionField>> parent)
  : shape_{std::move(shape)},
    field_size_{field_size},
    lr_{std::move(lr)},
    fid_{fid},
    parent_{std::move(parent)},
    physical_state_{make_internal_shared<PhysicalState>()}
{
}

inline const Legion::LogicalRegion& LogicalRegionField::region() const { return lr_; }

inline Legion::FieldID LogicalRegionField::field_id() const { return fid_; }

inline const std::optional<InternalSharedPtr<LogicalRegionField>>& LogicalRegionField::parent()
  const
{
  return parent_;
}

inline void LogicalRegionField::mark_attached() { attached_ = true; }

}  // namespace legate::detail
