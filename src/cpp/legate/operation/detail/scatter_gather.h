/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/logical_store.h>
#include <legate/operation/detail/operation.h>
#include <legate/partitioning/constraint.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <optional>

namespace legate::detail {

class ConstraintSolver;

class ScatterGather final : public Operation {
 public:
  ScatterGather(InternalSharedPtr<LogicalStore> target,
                InternalSharedPtr<LogicalStore> target_indirect,
                InternalSharedPtr<LogicalStore> source,
                InternalSharedPtr<LogicalStore> source_indirect,
                std::uint64_t unique_id,
                std::int32_t priority,
                mapping::detail::Machine machine,
                std::optional<std::int32_t> redop_kind);

  void set_source_indirect_out_of_range(bool flag);
  void set_target_indirect_out_of_range(bool flag);

  void validate() override;
  void launch(detail::Strategy* strategy) override;

  void add_to_solver(detail::ConstraintSolver& solver) override;

  [[nodiscard]] Kind kind() const override;
  [[nodiscard]] bool needs_flush() const override;

 private:
  bool source_indirect_out_of_range_{true};
  bool target_indirect_out_of_range_{true};
  StoreArg target_{};
  StoreArg target_indirect_{};
  StoreArg source_{};
  StoreArg source_indirect_{};
  InternalSharedPtr<Alignment> constraint_{};
  std::optional<std::int32_t> redop_kind_{};
};

}  // namespace legate::detail

#include <legate/operation/detail/scatter_gather.inl>
