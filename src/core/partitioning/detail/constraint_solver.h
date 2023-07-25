/* Copyright 2021 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
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
  void add_constraint(const Constraint* constraint);

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
  std::vector<const Constraint*> constraints_{};

 private:
  class EquivClass;
  std::map<const Variable, EquivClass*> equiv_class_map_{};
  std::vector<EquivClass*> equiv_classes_{};

 private:
  std::map<const Variable, bool> is_dependent_{};
};

}  // namespace legate::detail
