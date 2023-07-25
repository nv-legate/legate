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

#include <memory>

#include "core/utilities/typedefs.h"

namespace legate::detail {

class LogicalRegionField;
class Runtime;

class FieldManager {
 public:
  FieldManager(Runtime* runtime, const Domain& shape, uint32_t field_size);

 public:
  std::shared_ptr<LogicalRegionField> allocate_field();
  std::shared_ptr<LogicalRegionField> import_field(const Legion::LogicalRegion& region,
                                                   Legion::FieldID field_id);

 private:
  Runtime* runtime_;
  Domain shape_;
  uint32_t field_size_;

 private:
  using FreeField = std::pair<Legion::LogicalRegion, Legion::FieldID>;
  std::deque<FreeField> free_fields_;
};

}  // namespace legate::detail
