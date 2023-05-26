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
#include "core/partitioning/constraint.h"
#include "core/partitioning/restriction.h"
#include "core/utilities/ordered_set.h"

namespace legate {

struct ConstraintSolver {
 public:
  ConstraintSolver();
  ~ConstraintSolver();

 public:
  void add_partition_symbol(const Variable* partition_symbol);
  void add_constraint(const Constraint* constraint);

 public:
  void dump();

 public:
  const std::vector<const Variable*>& partition_symbols() const;
  const std::vector<const Constraint*>& constraints() const;

 public:
  void solve_constraints();
  const std::vector<const Variable*>& find_equivalence_class(
    const Variable* partition_symbol) const;
  const Restrictions& find_restrictions(const Variable* partition_symbol) const;

 private:
  ordered_set<const Variable*> partition_symbols_{};
  std::vector<const Constraint*> constraints_{};

 private:
  class EquivClass;
  std::map<const Variable, EquivClass*> equiv_class_map_{};
  std::vector<std::unique_ptr<EquivClass>> equiv_classes_{};
};

}  // namespace legate
