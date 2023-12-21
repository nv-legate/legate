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

#include "core/partitioning/detail/constraint.h"
#include "core/partitioning/restriction.h"
#include "core/utilities/detail/ordered_set.h"
#include "core/utilities/hash.h"
#include "core/utilities/internal_shared_ptr.h"

#include <unordered_map>
#include <vector>

namespace legate::detail {

class Strategy;

enum class IsOutput : bool {
  Y = true,
  N = false,
};

struct ConstraintSolver {
 public:
  ~ConstraintSolver();

  void add_partition_symbol(const Variable* partition_symbol, IsOutput is_output);
  void add_constraint(InternalSharedPtr<Constraint> constraint);

  void dump();

  [[nodiscard]] const std::vector<const Variable*>& partition_symbols() const;

  void solve_constraints();
  void solve_dependent_constraints(Strategy& strategy);
  [[nodiscard]] const std::vector<const Variable*>& find_equivalence_class(
    const Variable* partition_symbol) const;
  [[nodiscard]] const Restrictions& find_restrictions(const Variable* partition_symbol) const;
  [[nodiscard]] bool is_output(const Variable& partition_symbol) const;
  [[nodiscard]] bool is_dependent(const Variable& partition_symbol) const;

 private:
  ordered_set<const Variable*> partition_symbols_{};
  std::unordered_map<const Variable, bool> is_output_{};
  std::vector<InternalSharedPtr<Constraint>> constraints_{};

  struct EquivClass;
  std::unordered_map<const Variable, EquivClass*> equiv_class_map_{};
  std::vector<EquivClass*> equiv_classes_{};

  std::unordered_map<const Variable, bool> is_dependent_{};
};

}  // namespace legate::detail
