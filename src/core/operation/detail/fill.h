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

#include "core/operation/detail/operation.h"

namespace legate::detail {

class Fill : public Operation {
 private:
  friend class Runtime;
  Fill(std::shared_ptr<LogicalStore>&& lhs,
       std::shared_ptr<LogicalStore>&& value,
       int64_t unique_id,
       mapping::detail::Machine&& machine);

 public:
  void validate() override;
  void launch(Strategy* strategy) override;

 public:
  std::string to_string() const override;

 public:
  void add_to_solver(ConstraintSolver& solver) override;

 private:
  const Variable* lhs_var_;
  std::shared_ptr<LogicalStore> lhs_;
  std::shared_ptr<LogicalStore> value_;
};

}  // namespace legate::detail
