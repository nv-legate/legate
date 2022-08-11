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

#include <sstream>

#include "legion.h"

#include "core/partitioning/constraint.h"
#include "core/partitioning/constraint_graph.h"

namespace legate {

extern Legion::Logger log_legate;

void ConstraintGraph::add_partition_symbol(const Variable* partition_symbol)
{
  partition_symbols_.insert(partition_symbol);
}

void ConstraintGraph::add_constraint(const Constraint* constraint)
{
  constraints_.push_back(constraint);
}

void ConstraintGraph::compute_equivalence_classes()
{
  for (auto& part_symb : partition_symbols()) {
    all_equiv_class_entries_.emplace_back(new EquivClass(part_symb));
    equiv_classes_.insert({*part_symb, all_equiv_class_entries_.back().get()});
  }

  for (auto& constraint : constraints_) {
    auto* alignment = constraint->as_alignment();
    if (nullptr == alignment) continue;
#ifdef DEBUG_LEGATE
    assert(alignment->lhs()->as_variable() != nullptr &&
           alignment->rhs()->as_variable() != nullptr);
#endif

    std::vector<const Variable*> part_symbs_to_unify;
    alignment->find_partition_symbols(part_symbs_to_unify);
#ifdef DEBUG_LEGATE
    assert(!part_symbs_to_unify.empty());
#endif

    EquivClass* equiv_class = equiv_classes_[*part_symbs_to_unify[0]];
#ifdef DEBUG_LEGATE
    assert(equiv_class != nullptr);
#endif
    auto update_table = [&](auto* new_cls, auto* orig_cls) {
      while (orig_cls != nullptr) {
        equiv_classes_[*orig_cls->partition_symbol] = new_cls;
        orig_cls                                    = orig_cls->next;
      }
    };
    for (size_t idx = 1; idx < part_symbs_to_unify.size(); ++idx) {
      auto class_to_unify = equiv_classes_[*part_symbs_to_unify[idx]];
      auto result         = equiv_class->unify(class_to_unify);

      if (result != equiv_class) update_table(result, equiv_class);
      if (result != class_to_unify) update_table(result, class_to_unify);
      equiv_class = result;
    }
  }
}

void ConstraintGraph::find_equivalence_class(const Variable* partition_symbol,
                                             std::vector<const Variable*>& out_equiv_class) const
{
  auto finder = equiv_classes_.find(*partition_symbol);
#ifdef DEBUG_LEGATE
  assert(finder != equiv_classes_.end());
#endif
  auto equiv_class = finder->second;
  out_equiv_class.reserve(equiv_class->size);
  while (equiv_class != nullptr) {
    out_equiv_class.push_back(equiv_class->partition_symbol);
    equiv_class = equiv_class->next;
  }
}

void ConstraintGraph::dump()
{
  log_legate.debug("===== Constraint Graph =====");
  log_legate.debug() << "Variables:";
  for (auto& symbol : partition_symbols_.elements())
    log_legate.debug() << "  " << symbol->to_string();
  log_legate.debug() << "Constraints:";
  for (auto& constraint : constraints_) log_legate.debug() << "  " << constraint->to_string();
  log_legate.debug("============================");
}

const std::vector<const Variable*>& ConstraintGraph::partition_symbols() const
{
  return partition_symbols_.elements();
}

const std::vector<const Constraint*>& ConstraintGraph::constraints() const { return constraints_; }

}  // namespace legate
