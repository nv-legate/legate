/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/logical_store.h>
#include <legate/operation/detail/operation.h>
#include <legate/partitioning/detail/constraint.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <optional>

namespace legate::detail {

class ConstraintSolver;

class Copy final : public Operation {
 public:
  Copy(InternalSharedPtr<LogicalStore> target,
       InternalSharedPtr<LogicalStore> source,
       std::uint64_t unique_id,
       std::int32_t priority,
       mapping::detail::Machine machine,
       std::optional<std::int32_t> redop_kind);

  void validate() override;
  void launch(Strategy* strategy) override;

  void add_to_solver(ConstraintSolver& solver) override;

  [[nodiscard]] Kind kind() const override;
  [[nodiscard]] bool needs_flush() const override;

  /**
   * @return `true`.
   */
  [[nodiscard]] bool needs_partitioning() const override;

 private:
  StoreArg target_{};
  StoreArg source_{};
  InternalSharedPtr<Alignment> constraint_{};
  std::optional<std::int32_t> redop_kind_{};
};

}  // namespace legate::detail

#include <legate/operation/detail/copy.inl>
