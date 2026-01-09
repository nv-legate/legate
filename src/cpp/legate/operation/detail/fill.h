/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/detail/operation.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <variant>

namespace legate::detail {

class Fill final : public Operation {
 public:
  Fill(InternalSharedPtr<LogicalStore> lhs,
       InternalSharedPtr<LogicalStore> value,
       std::uint64_t unique_id,
       std::int32_t priority,
       mapping::detail::Machine machine);
  Fill(InternalSharedPtr<LogicalStore> lhs,
       Scalar value,
       std::uint64_t unique_id,
       std::int32_t priority,
       mapping::detail::Machine machine);

  void validate() override;
  void launch(Strategy* strategy) override;

  void add_to_solver(ConstraintSolver& solver) override;

  [[nodiscard]] Kind kind() const override;
  [[nodiscard]] bool needs_flush() const override;

  /**
   * @return `true`, `Fill` operations operate on specific subsets of stores.
   */
  [[nodiscard]] bool needs_partitioning() const override;

 private:
  Legion::Future get_fill_value_() const;

  const Variable* lhs_var_{};
  InternalSharedPtr<LogicalStore> lhs_{};
  std::variant<InternalSharedPtr<LogicalStore>, Scalar> value_{};
};

}  // namespace legate::detail

#include <legate/operation/detail/fill.inl>
