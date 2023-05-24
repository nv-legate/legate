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

#include "core/data/logical_store_detail.h"
#include "core/partitioning/constraint.h"
#include "core/partitioning/constraint_graph.h"
#include "core/runtime/operation.h"

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
    // TODO: partition symbols can be independent of any stores of the operation
    //       (e.g., when a symbol subsumes a union of two other symbols)
    int32_t ndim = part_symb->operation()->find_store(part_symb)->dim();
    all_equiv_class_entries_.emplace_back(new EquivClass(part_symb, ndim));
    equiv_classes_.insert({*part_symb, all_equiv_class_entries_.back().get()});
  }

  auto update_table = [&](auto* new_cls, auto* orig_cls) {
    while (orig_cls != nullptr) {
      equiv_classes_[*orig_cls->partition_symbol] = new_cls;
      orig_cls                                    = orig_cls->next;
    }
  };

  auto handle_alignment = [&](const Alignment* alignment) {
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
    for (size_t idx = 1; idx < part_symbs_to_unify.size(); ++idx) {
      auto class_to_unify = equiv_classes_[*part_symbs_to_unify[idx]];
      auto result         = equiv_class->unify(class_to_unify);

      if (result != equiv_class) update_table(result, equiv_class);
      if (result != class_to_unify) update_table(result, class_to_unify);
      equiv_class = result;
    }
  };

  auto handle_broadcast = [&](const Broadcast* broadcast) {
    auto* variable          = broadcast->variable();
    auto& axes              = broadcast->axes();
    EquivClass* equiv_class = equiv_classes_.at(*variable);
    for (uint32_t idx = 0; idx < axes.size(); ++idx) {
      uint32_t axis = axes[idx];
      // TODO: We want to check the axis eagerly and raise an exception
      // if it is out of bounds
      if (axis >= equiv_class->restrictions.size()) continue;
      equiv_class->restrictions[axes[idx]] = Restriction::FORBID;
    }
  };

  for (auto& constraint : constraints_) switch (constraint->kind()) {
      case Constraint::Kind::ALIGNMENT: {
        handle_alignment(constraint->as_alignment());
        break;
      }
      case Constraint::Kind::BROADCAST: {
        handle_broadcast(constraint->as_broadcast());
        break;
      }
    }
}

std::vector<const Variable*> ConstraintGraph::find_equivalence_class(
  const Variable* partition_symbol) const
{
  auto equiv_class = equiv_classes_.at(*partition_symbol);
  std::vector<const Variable*> result;
  result.reserve(equiv_class->size);
  while (equiv_class != nullptr) {
    result.push_back(equiv_class->partition_symbol);
    equiv_class = equiv_class->next;
  }
  return result;
}

Restrictions ConstraintGraph::find_restrictions(const Variable* partition_symbol) const
{
  auto equiv_class    = equiv_classes_.at(*partition_symbol);
  Restrictions result = equiv_class->restrictions;
  equiv_class         = equiv_class->next;
  while (equiv_class != nullptr) {
    join_inplace(result, equiv_class->restrictions);
    equiv_class = equiv_class->next;
  }
  return result;
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
