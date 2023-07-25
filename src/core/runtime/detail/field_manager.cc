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

#include "core/runtime/detail/field_manager.h"

#include "core/data/detail/logical_region_field.h"
#include "core/runtime/detail/region_manager.h"
#include "core/runtime/detail/runtime.h"

namespace legate::detail {

FieldManager::FieldManager(Runtime* runtime, const Domain& shape, uint32_t field_size)
  : runtime_(runtime), shape_(shape), field_size_(field_size)
{
}

std::shared_ptr<LogicalRegionField> FieldManager::allocate_field()
{
  LogicalRegionField* rf = nullptr;
  if (!free_fields_.empty()) {
    auto field = free_fields_.front();
    log_legate.debug("Field %u recycled in field manager %p", field.second, this);
    free_fields_.pop_front();
    rf = new LogicalRegionField(field.first, field.second);
  } else {
    auto rgn_mgr   = runtime_->find_or_create_region_manager(shape_);
    auto [lr, fid] = rgn_mgr->allocate_field(field_size_);
    rf             = new LogicalRegionField(lr, fid);
    log_legate.debug("Field %u created in field manager %p", fid, this);
  }
  assert(rf != nullptr);
  return std::shared_ptr<LogicalRegionField>(rf, [this](auto* field) {
    log_legate.debug("Field %u freed in field manager %p", field->field_id(), this);
    this->free_fields_.push_back(FreeField(field->region(), field->field_id()));
    delete field;
  });
}

std::shared_ptr<LogicalRegionField> FieldManager::import_field(const Legion::LogicalRegion& region,
                                                               Legion::FieldID field_id)
{
  // Import the region only if the region manager is created fresh
  auto rgn_mgr = runtime_->find_or_create_region_manager(shape_);
  if (!rgn_mgr->has_space()) rgn_mgr->import_region(region);

  log_legate.debug("Field %u imported in field manager %p", field_id, this);

  auto* rf = new LogicalRegionField(region, field_id);
  return std::shared_ptr<LogicalRegionField>(rf, [this](auto* field) {
    log_legate.debug("Field %u freed in field manager %p", field->field_id(), this);
    this->free_fields_.push_back(FreeField(field->region(), field->field_id()));
    delete field;
  });
}

}  // namespace legate::detail
