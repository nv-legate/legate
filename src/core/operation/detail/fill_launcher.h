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

#include "core/mapping/detail/machine.h"

namespace legate::detail {

class LogicalStore;
struct ProjectionInfo;
class BufferBuilder;

class FillLauncher {
 public:
  FillLauncher(const mapping::detail::Machine& machine, int64_t tag = 0);

  void launch(const Legion::Domain& launch_domain,
              LogicalStore* lhs,
              const ProjectionInfo& lhs_proj,
              LogicalStore* value);
  void launch_single(LogicalStore* lhs, const ProjectionInfo& lhs_proj, LogicalStore* value);

 private:
  void pack_mapper_arg(BufferBuilder& buffer, Legion::ProjectionID proj_id);

  const mapping::detail::Machine& machine_;
  int64_t tag_;
};

}  // namespace legate::detail

#include "core/operation/detail/fill_launcher.inl"
