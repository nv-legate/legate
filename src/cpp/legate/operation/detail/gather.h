/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
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

class Gather final : public Operation {
 public:
  Gather(InternalSharedPtr<LogicalStore> target,
         InternalSharedPtr<LogicalStore> source,
         InternalSharedPtr<LogicalStore> source_indirect,
         std::uint64_t unique_id,
         std::int32_t priority,
         mapping::detail::Machine machine,
         std::optional<std::int32_t> redop_kind);

  void set_indirect_out_of_range(bool flag);

  void validate() override;
  void launch(Strategy* strategy) override;

  void add_to_solver(ConstraintSolver& solver) override;

  [[nodiscard]] Kind kind() const override;
  [[nodiscard]] bool needs_flush() const override;

  /**
   * @return `true`, `Gather` operations operate on specific subsets of stores.
   */
  [[nodiscard]] bool needs_partitioning() const override;

 private:
  bool out_of_range_{true};
  StoreArg target_{};
  StoreArg source_{};
  StoreArg source_indirect_{};
  InternalSharedPtr<Constraint> constraint_{};
  std::optional<std::int32_t> redop_kind_{};
};

}  // namespace legate::detail

#include <legate/operation/detail/gather.inl>
