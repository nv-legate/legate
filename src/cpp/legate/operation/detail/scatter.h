/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

class Scatter final : public Operation {
 public:
  Scatter(InternalSharedPtr<LogicalStore> target,
          InternalSharedPtr<LogicalStore> target_indirect,
          InternalSharedPtr<LogicalStore> source,
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

 private:
  bool out_of_range_{true};
  StoreArg target_{};
  StoreArg target_indirect_{};
  StoreArg source_{};
  InternalSharedPtr<Alignment> constraint_{};
  std::optional<std::int32_t> redop_kind_{};
};

}  // namespace legate::detail

#include <legate/operation/detail/scatter.inl>
