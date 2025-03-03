/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/detail/constraint_solver.h>

namespace legate::detail {

inline void ConstraintSolver::add_constraint(InternalSharedPtr<Constraint> constraint)
{
  constraints_.push_back(std::move(constraint));
}

inline AccessMode ConstraintSolver::find_access_mode(const Variable& partition_symbol) const
{
  return access_modes_.at(partition_symbol);
}

inline bool ConstraintSolver::is_output(const Variable& partition_symbol) const
{
  return find_access_mode(partition_symbol) == AccessMode::WRITE;
}

inline bool ConstraintSolver::is_dependent(const Variable& partition_symbol) const
{
  return is_dependent_.at(partition_symbol);
}

inline const std::vector<const Variable*>& ConstraintSolver::partition_symbols() const
{
  return partition_symbols_.elements();
}

}  // namespace legate::detail
