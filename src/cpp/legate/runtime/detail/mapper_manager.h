/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/zstring_view.h>

#include <legion.h>
#include <legion/api/mapping.h>
#include <legion/api/types.h>

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

#include <legate/runtime/detail/mapper_manager.inl>
