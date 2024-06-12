/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

class ScatterGather final : public Operation {
 public:
  ScatterGather(InternalSharedPtr<LogicalStore> target,
                InternalSharedPtr<LogicalStore> target_indirect,
                InternalSharedPtr<LogicalStore> source,
                InternalSharedPtr<LogicalStore> source_indirect,
                std::uint64_t unique_id,
                std::int32_t priority,
                mapping::detail::Machine machine,
                std::optional<std::int32_t> redop);

  void set_source_indirect_out_of_range(bool flag);
  void set_target_indirect_out_of_range(bool flag);

  void validate() override;
  void launch(detail::Strategy* strategy) override;

  void add_to_solver(detail::ConstraintSolver& solver) override;

  [[nodiscard]] Kind kind() const override;

 private:
  bool source_indirect_out_of_range_{true};
  bool target_indirect_out_of_range_{true};
  StoreArg target_{};
  StoreArg target_indirect_{};
  StoreArg source_{};
  StoreArg source_indirect_{};
  InternalSharedPtr<Alignment> constraint_{};
  std::optional<std::int32_t> redop_{};
};

}  // namespace legate::detail

#include "core/operation/detail/scatter_gather.inl"
