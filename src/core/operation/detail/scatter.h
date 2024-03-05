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

#include "core/data/detail/logical_store.h"
#include "core/operation/detail/operation.h"
#include "core/partitioning/constraint.h"
#include "core/utilities/internal_shared_ptr.h"

#include <optional>
#include <string>

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
          std::optional<std::int32_t> redop);

  void set_indirect_out_of_range(bool flag);

  void validate() override;
  void launch(Strategy* strategy) override;

  void add_to_solver(ConstraintSolver& solver) override;

  [[nodiscard]] std::string to_string() const override;

 private:
  bool out_of_range_{true};
  StoreArg target_;
  StoreArg target_indirect_;
  StoreArg source_;
  InternalSharedPtr<Constraint> constraint_;
  std::optional<std::int32_t> redop_;
};

}  // namespace legate::detail

#include "core/operation/detail/scatter.inl"
