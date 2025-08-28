/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/mapping/detail/machine.h>
#include <legate/operation/detail/launcher_arg.h>
#include <legate/operation/detail/store_projection.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/detail/small_vector.h>

namespace legate {

class Scalar;

}  // namespace legate

namespace legate::detail {

class LogicalStore;
class OutputRequirementAnalyzer;
class RequirementAnalyzer;
class BufferBuilder;

class CopyArg final : public Serializable {
 public:
  CopyArg(std::uint32_t req_idx,
          LogicalStore* store,
          Legion::FieldID field_id,
          Legion::PrivilegeMode privilege,
          StoreProjection store_proj);

  void pack(BufferBuilder& buffer) const override;

  /**
   * @brief Create a region requirement for an argument to a Copy operation.
   *
   * @tparam SINGLE Whether the argument will have exclusive access to the data or not.
   */
  template <bool SINGLE>
  [[nodiscard]] Legion::RegionRequirement create_requirement();

 private:
  std::uint32_t req_idx_{};
  LogicalStore* store_{};
  Legion::LogicalRegion region_{};
  Legion::FieldID field_id_{};
  Legion::PrivilegeMode privilege_{};
  StoreProjection store_proj_{};
};

class CopyLauncher {
 public:
  CopyLauncher(const mapping::detail::Machine& machine,
               std::int32_t priority,
               std::int64_t tag = 0);

  void add_input(const InternalSharedPtr<LogicalStore>& store, StoreProjection store_proj);
  void add_output(const InternalSharedPtr<LogicalStore>& store, StoreProjection store_proj);
  void add_inout(const InternalSharedPtr<LogicalStore>& store, StoreProjection store_proj);
  void add_reduction(const InternalSharedPtr<LogicalStore>& store, StoreProjection store_proj);
  void add_source_indirect(const InternalSharedPtr<LogicalStore>& store,
                           StoreProjection store_proj);
  void add_target_indirect(const InternalSharedPtr<LogicalStore>& store,
                           StoreProjection store_proj);

  void add_store(SmallVector<CopyArg>& args,
                 const InternalSharedPtr<LogicalStore>& store,
                 StoreProjection store_proj,
                 Legion::PrivilegeMode privilege);

  void set_source_indirect_out_of_range(bool flag) { source_indirect_out_of_range_ = flag; }

  void set_target_indirect_out_of_range(bool flag) { target_indirect_out_of_range_ = flag; }

  void execute(const Legion::Domain& launch_domain);
  void execute_single();

  void pack_args(BufferBuilder& buffer);
  void pack_sharding_functor_id(BufferBuilder& buffer);

 private:
  template <typename Launcher>
  void populate_copy_(Launcher& launcher);

  const mapping::detail::Machine& machine_;
  std::int32_t priority_{static_cast<std::int32_t>(TaskPriority::DEFAULT)};
  std::int64_t tag_{};
  Legion::ProjectionID key_proj_id_{};

  SmallVector<CopyArg> inputs_{};
  SmallVector<CopyArg> outputs_{};
  SmallVector<CopyArg> source_indirect_{};
  SmallVector<CopyArg> target_indirect_{};

  bool source_indirect_out_of_range_{true};
  bool target_indirect_out_of_range_{true};
};

}  // namespace legate::detail

#include <legate/operation/detail/copy_launcher.inl>
