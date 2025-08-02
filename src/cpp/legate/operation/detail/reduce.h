/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/mapping/detail/machine.h>
#include <legate/operation/detail/operation.h>
#include <legate/utilities/internal_shared_ptr.h>

namespace legate::detail {

class Library;

class Reduce final : public Operation {
 public:
  Reduce(const Library& library,
         InternalSharedPtr<LogicalStore> store,
         InternalSharedPtr<LogicalStore> out_store,
         LocalTaskID task_id,
         std::uint64_t unique_id,
         std::int32_t radix,
         std::int32_t priority,
         mapping::detail::Machine machine);

  void launch(Strategy*) override;
  void validate() override;
  void add_to_solver(ConstraintSolver& solver) override;

  [[nodiscard]] Kind kind() const override;
  [[nodiscard]] bool needs_flush() const override;

  /**
   * @return `true`, `Reduce` operations require the operands to be pre-partitioned.
   */
  [[nodiscard]] bool needs_partitioning() const override;

 private:
  std::int32_t radix_{};
  std::reference_wrapper<const Library> library_;
  LocalTaskID task_id_{};
  InternalSharedPtr<LogicalStore> input_{};
  InternalSharedPtr<LogicalStore> output_{};
  const Variable* input_part_{};
  const Variable* output_part_{};
};

}  // namespace legate::detail

#include <legate/operation/detail/reduce.inl>
