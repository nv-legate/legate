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

#include "core/partitioning/detail/constraint_solver.h"

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
