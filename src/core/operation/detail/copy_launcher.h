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

#include <memory>
#include <vector>

namespace legate {
class Scalar;
}  // namespace legate

namespace legate::detail {

struct CopyArg;
class LogicalStore;
struct ProjectionInfo;
class OutputRequirementAnalyzer;
class RequirementAnalyzer;
class BufferBuilder;

class CopyLauncher {
 public:
  CopyLauncher(const mapping::detail::Machine& machine, int64_t tag = 0);

  ~CopyLauncher();

  void add_input(detail::LogicalStore* store, std::unique_ptr<ProjectionInfo> proj_info);
  void add_output(detail::LogicalStore* store, std::unique_ptr<ProjectionInfo> proj_info);
  void add_inout(detail::LogicalStore* store, std::unique_ptr<ProjectionInfo> proj_info);
  void add_reduction(detail::LogicalStore* store, std::unique_ptr<ProjectionInfo> proj_info);
  void add_source_indirect(detail::LogicalStore* store, std::unique_ptr<ProjectionInfo> proj_info);
  void add_target_indirect(detail::LogicalStore* store, std::unique_ptr<ProjectionInfo> proj_info);

  void add_store(std::vector<CopyArg*>& args,
                 detail::LogicalStore* store,
                 std::unique_ptr<ProjectionInfo> proj_info,
                 Legion::PrivilegeMode privilege);

  void set_source_indirect_out_of_range(bool flag) { source_indirect_out_of_range_ = flag; }
  void set_target_indirect_out_of_range(bool flag) { target_indirect_out_of_range_ = flag; }

  void execute(const Legion::Domain& launch_domain);
  void execute_single();

  void pack_args(BufferBuilder& buffer);
  void pack_sharding_functor_id(BufferBuilder& buffer);
  template <class Launcher>
  void populate_copy(Launcher& launcher);

 private:
  const mapping::detail::Machine& machine_;
  int64_t tag_{};
  Legion::ProjectionID key_proj_id_{};

  std::vector<CopyArg*> inputs_{};
  std::vector<CopyArg*> outputs_{};
  std::vector<CopyArg*> source_indirect_{};
  std::vector<CopyArg*> target_indirect_{};

  bool source_indirect_out_of_range_{true};
  bool target_indirect_out_of_range_{true};
};

}  // namespace legate::detail

#include "core/operation/detail/copy_launcher.inl"
