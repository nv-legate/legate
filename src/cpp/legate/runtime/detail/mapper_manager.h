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

#include "legate/utilities/detail/zstring_view.h"

#include <legion.h>
#include <legion/legion_mapping.h>
#include <legion/legion_types.h>

namespace legate::mapping::detail {

class BaseMapper;

}  // namespace legate::mapping::detail

namespace legate::detail {

class MapperManager {
 public:
  MapperManager();

  [[nodiscard]] Legion::MapperID mapper_id() const;

 private:
  explicit MapperManager(Legion::Runtime* legion_runtime);

  Legion::MapperID mapper_id_{};
};

}  // namespace legate::detail

#include "legate/runtime/detail/mapper_manager.inl"
