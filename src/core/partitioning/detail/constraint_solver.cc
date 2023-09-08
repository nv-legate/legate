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

#include "core/partitioning/detail/constraint_solver.h"

#include <sstream>

#include "legion.h"

#include "core/data/detail/logical_store.h"
#include "core/operation/detail/operation.h"
#include "core/partitioning/detail/partitioner.h"

namespace legate {
extern Legion::Logger log_legate;
}  // namespace legate

namespace legate::detail {

namespace {

struct UnionFindEntry {
  UnionFindEntry(const Variable* symb, const detail::LogicalStore* store)
    : partition_symbol(symb), restrictions(store->compute_restrictions()), next(nullptr), size(1)
  {
  }

  UnionFindEntry* unify(UnionFindEntry* other)
  {
    UnionFindEntry* self = this;
    if (self->size < other->size) std::swap(self, other);

    auto end = self;
    while (end->next != nullptr) end = end->next;
    end->next = other;
    end->size += other->size;
    return self;
  }

  const Variable* partition_symbol;
  Restrictions restrictions;
  UnionFindEntry* next;
  size_t size;
};

}  // namespace

struct ConstraintSolver::EquivClass {
  EquivClass(const UnionFindEntry* entry)
  {
    partition_symbols.reserve(entry->size);
    partition_symbols.push_back(entry->partition_symbol);
    restrictions = entry->restrictions;

    auto* next = entry->next;
    while (next != nullptr) {
      partition_symbols.push_back(next->partition_symbol);
      join_inplace(restrictions, next->restrictions);
      next = next->next;
    }
  }

  std::vector<const Variable*> partition_symbols;
  Restrictions restrictions;
};

ConstraintSolver::ConstraintSolver() {}

ConstraintSolver::~ConstraintSolver()
{
  for (auto equiv_class : equiv_classes_) delete equiv_class;
}

void ConstraintSolver::add_partition_symbol(const Variable* partition_symbol, bool is_output)
{
  partition_symbols_.insert(partition_symbol);
  is_output_.insert({*partition_symbol, is_output});
  is_dependent_.insert({*partition_symbol, false});
}

void ConstraintSolver::add_constraint(std::unique_ptr<Constraint> constraint)
{
  constraints_.push_back(std::move(constraint));
}

void ConstraintSolver::solve_constraints()
{
  std::vector<UnionFindEntry> entries;
  std::map<const Variable, UnionFindEntry*> table;

  // Initialize the table by creating singleton equivalence classes
  const auto& all_symbols = partition_symbols();
  entries.reserve(all_symbols.size());

  auto initialize = [&entries, &table](const auto& all_symbols) {
    for (auto& symb : all_symbols) {
      // TODO: partition symbols can be independent of any stores of the operation
      //       (e.g., when a symbol subsumes a union of two other symbols)
      auto store = symb->operation()->find_store(symb);
      entries.emplace_back(symb, store.get());
      table.insert({*symb, &entries.back()});
    }
  };
  initialize(all_symbols);

  // Unify equivalence classes based on alignment constraints
  auto handle_alignment = [&table](const Alignment* alignment) {
    auto update_table = [&table](UnionFindEntry* old_cls, UnionFindEntry* new_cls) {
      while (old_cls != nullptr) {
        table[*old_cls->partition_symbol] = new_cls;
        old_cls                           = old_cls->next;
      }
    };

    std::vector<const Variable*> part_symbs_to_unify;
    alignment->find_partition_symbols(part_symbs_to_unify);
#ifdef DEBUG_LEGATE
    assert(!part_symbs_to_unify.empty());
#endif

    auto it           = part_symbs_to_unify.begin();
    auto* equiv_class = table[**it++];
#ifdef DEBUG_LEGATE
    assert(equiv_class != nullptr);
#endif
    for (; it != part_symbs_to_unify.end(); ++it) {
      auto* to_unify = table[**it];
      auto* result   = equiv_class->unify(to_unify);

      if (result != equiv_class) update_table(equiv_class, result);
      if (result != to_unify) update_table(to_unify, result);
    }
  };

  // Set the restrictions according to broadcasting constraints
  auto handle_broadcast = [&table](const Broadcast* broadcast) {
    auto* variable    = broadcast->variable();
    auto& axes        = broadcast->axes();
    auto* equiv_class = table.at(*variable);
    for (uint32_t idx = 0; idx < axes.size(); ++idx) {
      uint32_t axis = axes[idx];
      // TODO: We want to check the axis eagerly and raise an exception
      // if it is out of bounds
      if (axis >= equiv_class->restrictions.size()) continue;
      equiv_class->restrictions[axes[idx]] = Restriction::FORBID;
    }
  };

  // Here we only mark dependent partition symbols
  auto handle_image_constraint = [&](const ImageConstraint* image_constraint) {
    is_dependent_[*image_constraint->var_range()] = true;
  };
  auto handle_scale_constraint = [&](const ScaleConstraint* scale_constraint) {
    is_dependent_[*scale_constraint->var_bigger()] = true;
  };

  // Reflect each constraint to the solver state
  for (auto& constraint : constraints_) {
    switch (constraint->kind()) {
      case Constraint::Kind::ALIGNMENT: {
        auto* alignment = constraint->as_alignment();
        if (!alignment->is_trivial()) handle_alignment(alignment);
        break;
      }
      case Constraint::Kind::BROADCAST: {
        handle_broadcast(constraint->as_broadcast());
        break;
      }
      case Constraint::Kind::IMAGE: {
        handle_image_constraint(constraint->as_image_constraint());
        break;
      }
      case Constraint::Kind::SCALE: {
        handle_scale_constraint(constraint->as_scale_constraint());
        break;
      }
    }
  }

  // Combine states of each union of equivalence classes into one
  std::unordered_set<UnionFindEntry*> distinct_entries;
  for (auto& [_, entry] : table) distinct_entries.insert(entry);

  for (auto* entry : distinct_entries) {
    auto equiv_class = new EquivClass(entry);
    for (auto* symb : equiv_class->partition_symbols) equiv_class_map_.insert({*symb, equiv_class});
    equiv_classes_.push_back(equiv_class);
  }
}

void ConstraintSolver::solve_dependent_constraints(Strategy& strategy)
{
  auto solve_image_constraint = [&strategy](const ImageConstraint* image_constraint) {
    auto image = image_constraint->resolve(strategy);
    strategy.insert(image_constraint->var_range(), std::move(image));
  };

  auto solve_scale_constraint = [&strategy](const ScaleConstraint* scale_constraint) {
    auto scaled = scale_constraint->resolve(strategy);
    strategy.insert(scale_constraint->var_bigger(), std::move(scaled));
  };

  for (auto& constraint : constraints_) {
    switch (constraint->kind()) {
      case Constraint::Kind::IMAGE: {
        solve_image_constraint(constraint->as_image_constraint());
        break;
      }
      case Constraint::Kind::SCALE: {
        solve_scale_constraint(constraint->as_scale_constraint());
        break;
      }
      default: {
        continue;
      }
    }
  }
}

const std::vector<const Variable*>& ConstraintSolver::find_equivalence_class(
  const Variable* partition_symbol) const
{
  return equiv_class_map_.at(*partition_symbol)->partition_symbols;
}

const Restrictions& ConstraintSolver::find_restrictions(const Variable* partition_symbol) const
{
  return equiv_class_map_.at(*partition_symbol)->restrictions;
}

bool ConstraintSolver::is_output(const Variable& partition_symbol) const
{
  return is_output_.at(partition_symbol);
}

bool ConstraintSolver::is_dependent(const Variable& partition_symbol) const
{
  return is_dependent_.at(partition_symbol);
}

void ConstraintSolver::dump()
{
  log_legate.debug("===== Constraint Graph =====");
  log_legate.debug() << "Stores:";
  for (auto& symbol : partition_symbols_.elements()) {
    auto store = symbol->operation()->find_store(symbol);
    log_legate.debug() << "  " << symbol->to_string() << " ~> " << store->to_string();
  }
  log_legate.debug() << "Variables:";
  for (auto& symbol : partition_symbols_.elements())
    log_legate.debug() << "  " << symbol->to_string();
  log_legate.debug() << "Constraints:";
  for (auto& constraint : constraints_) log_legate.debug() << "  " << constraint->to_string();
  log_legate.debug("============================");
}

const std::vector<const Variable*>& ConstraintSolver::partition_symbols() const
{
  return partition_symbols_.elements();
}

}  // namespace legate::detail
