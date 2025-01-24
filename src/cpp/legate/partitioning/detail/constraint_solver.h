/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <legate/partitioning/detail/constraint.h>
#include <legate/partitioning/detail/restriction.h>
#include <legate/utilities/detail/ordered_set.h>
#include <legate/utilities/hash.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>

namespace legate::detail {

class Strategy;

enum class AccessMode : std::uint8_t {
  READ   = 0,
  REDUCE = 1,
  WRITE  = 2,
};

class ConstraintSolver {
 public:
  ~ConstraintSolver();

  void add_partition_symbol(const Variable* partition_symbol, AccessMode access_mode);
  void add_constraint(InternalSharedPtr<Constraint> constraint);

  void dump();

  [[nodiscard]] const std::vector<const Variable*>& partition_symbols() const;

  void solve_constraints();
  void solve_dependent_constraints(Strategy& strategy);
  [[nodiscard]] const std::vector<const Variable*>& find_equivalence_class(
    const Variable* partition_symbol) const;
  [[nodiscard]] const Restrictions& find_restrictions(const Variable* partition_symbol) const;
  [[nodiscard]] AccessMode find_access_mode(const Variable& partition_symbol) const;
  [[nodiscard]] bool is_output(const Variable& partition_symbol) const;
  [[nodiscard]] bool is_dependent(const Variable& partition_symbol) const;

 private:
  ordered_set<const Variable*> partition_symbols_{};
  std::unordered_map<Variable, AccessMode> access_modes_{};
  std::vector<InternalSharedPtr<Constraint>> constraints_{};

  class UnionFindEntry;

  class EquivClass {
   public:
    explicit EquivClass(const UnionFindEntry* entry);

    std::vector<const Variable*> partition_symbols{};
    Restrictions restrictions{};
  };

  std::unordered_map<Variable, EquivClass*> equiv_class_map_{};
  std::vector<std::unique_ptr<EquivClass>> equiv_classes_{};

  std::unordered_map<Variable, bool> is_dependent_{};
};

}  // namespace legate::detail

#include <legate/partitioning/detail/constraint_solver.inl>
