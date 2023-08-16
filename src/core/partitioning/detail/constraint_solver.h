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

#include <list>
#include <map>
#include <memory>
#include <vector>
#include "core/partitioning/detail/constraint.h"
#include "core/partitioning/restriction.h"
#include "core/utilities/ordered_set.h"

namespace legate::detail {

class Strategy;

struct ConstraintSolver {
 public:
  ConstraintSolver();
  ~ConstraintSolver();

 public:
  void add_partition_symbol(const Variable* partition_symbol, bool is_output = false);
  void add_constraint(std::unique_ptr<Constraint> constraint);

 public:
  void dump();

 public:
  const std::vector<const Variable*>& partition_symbols() const;

 public:
  void solve_constraints();
  void solve_dependent_constraints(Strategy& strategy);
  const std::vector<const Variable*>& find_equivalence_class(
    const Variable* partition_symbol) const;
  const Restrictions& find_restrictions(const Variable* partition_symbol) const;
  bool is_output(const Variable& partition_symbol) const;
  bool is_dependent(const Variable& partition_symbol) const;

 private:
  ordered_set<const Variable*> partition_symbols_{};
  std::map<const Variable, bool> is_output_{};
  std::vector<std::unique_ptr<Constraint>> constraints_{};

 private:
  class EquivClass;
  std::map<const Variable, EquivClass*> equiv_class_map_{};
  std::vector<EquivClass*> equiv_classes_{};

 private:
  std::map<const Variable, bool> is_dependent_{};
};

}  // namespace legate::detail
