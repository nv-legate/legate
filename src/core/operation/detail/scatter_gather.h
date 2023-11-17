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

#include <memory>
#include <optional>
#include <string>

namespace legate::detail {

struct ConstraintSolver;

class ScatterGather final : public Operation {
 public:
  ScatterGather(std::shared_ptr<LogicalStore> target,
                std::shared_ptr<LogicalStore> target_indirect,
                std::shared_ptr<LogicalStore> source,
                std::shared_ptr<LogicalStore> source_indirect,
                uint64_t unique_id,
                mapping::detail::Machine&& machine,
                std::optional<int32_t> redop);

  void set_source_indirect_out_of_range(bool flag);
  void set_target_indirect_out_of_range(bool flag);

  void validate() override;
  void launch(detail::Strategy* strategy) override;

  void add_to_solver(detail::ConstraintSolver& solver) override;

  [[nodiscard]] std::string to_string() const override;

 private:
  bool source_indirect_out_of_range_{true};
  bool target_indirect_out_of_range_{true};
  StoreArg target_;
  StoreArg target_indirect_;
  StoreArg source_;
  StoreArg source_indirect_;
  std::shared_ptr<Constraint> constraint_;
  std::optional<int32_t> redop_;
};

}  // namespace legate::detail

#include "core/operation/detail/scatter_gather.inl"
