/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/mapping/detail/machine.h>
#include <legate/operation/detail/store_projection.h>
#include <legate/utilities/detail/core_ids.h>

namespace legate::detail {

class LogicalStore;
class BufferBuilder;

class FillLauncher {
 public:
  FillLauncher(const mapping::detail::Machine& machine,
               std::int32_t priority,
               std::int64_t tag = 0);

  void launch(const Legion::Domain& launch_domain,
              LogicalStore* lhs,
              const StoreProjection& lhs_proj,
              Legion::Future value);
  void launch_single(LogicalStore* lhs, const StoreProjection& lhs_proj, Legion::Future value);

 private:
  [[nodiscard]] BufferBuilder pack_mapper_arg_(Legion::ProjectionID proj_id) const;

  const mapping::detail::Machine& machine_;
  std::int32_t priority_{static_cast<std::int32_t>(TaskPriority::DEFAULT)};
  std::int64_t tag_{};
};

}  // namespace legate::detail

#include <legate/operation/detail/fill_launcher.inl>
