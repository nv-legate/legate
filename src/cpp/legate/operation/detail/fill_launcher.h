/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/core_ids.h>

#include <cstdint>
#include <string_view>

namespace legate::mapping::detail {

class Machine;

}  // namespace legate::mapping::detail

namespace legate::detail {

class LogicalStore;
class BufferBuilder;
class StoreProjection;

class FillLauncher {
 public:
  FillLauncher(const mapping::detail::Machine& machine,
               std::int32_t priority,
               std::string_view provenance);

  void launch(const Legion::Domain& launch_domain,
              LogicalStore* lhs,
              const StoreProjection& lhs_proj,
              Legion::Future value);
  void launch_single(LogicalStore* lhs, const StoreProjection& lhs_proj, Legion::Future value);

 private:
  [[nodiscard]] BufferBuilder pack_mapper_arg_(Legion::ProjectionID proj_id) const;

  const mapping::detail::Machine& machine_;
  std::int32_t priority_{static_cast<std::int32_t>(TaskPriority::DEFAULT)};
  std::string_view provenance_{};
};

}  // namespace legate::detail

#include <legate/operation/detail/fill_launcher.inl>
