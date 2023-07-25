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

#include <memory>

#include "core/data/detail/logical_store.h"
#include "core/operation/detail/operation.h"
#include "core/partitioning/constraint.h"

namespace legate::detail {

class ConstraintSolver;

class Copy : public Operation {
 public:
  Copy(std::shared_ptr<LogicalStore> target,
       std::shared_ptr<LogicalStore> source,
       int64_t unique_id,
       mapping::detail::Machine&& machine);

 public:
  void validate() override;
  void launch(Strategy* strategy) override;

 public:
  void add_to_solver(ConstraintSolver& solver) override;

 public:
  std::string to_string() const override;

 private:
  StoreArg target_;
  StoreArg source_;
  std::unique_ptr<Constraint> constraint_;
};

}  // namespace legate::detail
