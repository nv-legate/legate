/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/constraint_solver.h>

#include <legate/operation/detail/operation.h>
#include <legate/partitioning/detail/partitioner.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/detail/small_vector.h>

#include <algorithm>
#include <cstddef>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace legate::detail {

class ConstraintSolver::UnionFindEntry {
 public:
  UnionFindEntry(const Variable* symb, Restrictions rs)
    : partition_symbol{symb}, restrictions{std::move(rs)}
  {
  }

  [[nodiscard]] UnionFindEntry* unify(UnionFindEntry* other)
  {
    if (this == other) {
      return this;
    }

    UnionFindEntry* self = this;
    if (self->size < other->size) {
      std::swap(self, other);
    }

    auto end = self;

    while (end->next) {
      end = end->next;
    }
    end->next = other;
    end->size += other->size;
    return self;
  }

  const Variable* partition_symbol{};
  Restrictions restrictions{};
  UnionFindEntry* next{};
  std::size_t size{1};
};

ConstraintSolver::EquivClass::EquivClass(const UnionFindEntry* entry)
{
  partition_symbols.reserve(entry->size);
  partition_symbols.emplace_back(entry->partition_symbol);
  restrictions = entry->restrictions;

  auto* next = entry->next;

  while (next) {
    partition_symbols.emplace_back(next->partition_symbol);
    restrictions.join_inplace(next->restrictions);
    next = next->next;
  }
}

ConstraintSolver::~ConstraintSolver() = default;

void ConstraintSolver::add_partition_symbol(const Variable* partition_symbol,
                                            AccessMode access_mode)
{
  partition_symbols_.insert(partition_symbol);

  const auto [it, inserted] = access_modes_.try_emplace(*partition_symbol, access_mode);

  if (!inserted) {
    it->second = std::max(it->second, access_mode);
  }
}

void ConstraintSolver::solve_constraints()
{
  std::vector<UnionFindEntry> entries;
  std::unordered_map<Variable, UnionFindEntry*> table;

  // Initialize the table by creating singleton equivalence classes
  const auto& all_symbols = partition_symbols();

  entries.reserve(all_symbols.size());
  for (auto&& symb : all_symbols) {
    // TODO(wonchanl): partition symbols can be independent of any stores of the operation
    // (e.g., when a symbol subsumes a union of two other symbols)
    auto store  = symb->operation()->find_store(symb);
    auto& entry = entries.emplace_back(symb, store->compute_restrictions(is_output(*symb)));
    table.insert({*symb, &entry});
    is_dependent_.insert({*symb, false});
  }

  // Unify equivalence classes based on alignment constraints
  const auto handle_alignment = [&table](const Alignment& alignment) {
    const auto update_table = [&table](UnionFindEntry* old_cls, UnionFindEntry* new_cls) {
      while (old_cls != nullptr) {
        table[*old_cls->partition_symbol] = new_cls;
        old_cls                           = old_cls->next;
      }
    };

    SmallVector<const Variable*> part_symbs_to_unify;

    alignment.find_partition_symbols(part_symbs_to_unify);
    LEGATE_ASSERT(!part_symbs_to_unify.empty());

    auto it                 = part_symbs_to_unify.begin();
    auto* const equiv_class = table[**it++];

    LEGATE_ASSERT(equiv_class != nullptr);
    for (; it != part_symbs_to_unify.end(); ++it) {
      auto* to_unify = table[**it];
      auto* result   = equiv_class->unify(to_unify);

      if (result != equiv_class) {
        update_table(equiv_class, result);
      }
      if (result != to_unify) {
        update_table(to_unify, result);
      }
    }
  };

  // Set the restrictions according to broadcasting constraints
  const auto handle_broadcast = [&table](const Broadcast& broadcast) {
    auto&& axes             = broadcast.axes();
    auto* const equiv_class = table.at(*broadcast.variable());

    if (axes.empty()) {
      equiv_class->restrictions.restrict_all_dimensions();
      return;
    }

    for (auto&& ax : axes) {
      equiv_class->restrictions.restrict_dimension(ax);
    }
  };

  // Here we only mark dependent partition symbols
  const auto handle_image_constraint = [&](const ImageConstraint& image_constraint) {
    is_dependent_[*image_constraint.var_range()] = true;
  };
  const auto handle_scale_constraint = [&](const ScaleConstraint& scale_constraint) {
    is_dependent_[*scale_constraint.var_bigger()] = true;
  };
  const auto handle_bloat_constraint = [&](const BloatConstraint& bloat_constraint) {
    is_dependent_[*bloat_constraint.var_bloat()] = true;
  };

  // Reflect each constraint to the solver state
  for (auto&& constraint : constraints_) {
    switch (constraint->kind()) {
      case Constraint::Kind::ALIGNMENT: {
        if (const auto& alignment = static_cast<const Alignment&>(*constraint);
            !alignment.is_trivial()) {
          handle_alignment(alignment);
        }
        break;
      }
      case Constraint::Kind::BROADCAST: {
        handle_broadcast(static_cast<const Broadcast&>(*constraint));
        break;
      }
      case Constraint::Kind::IMAGE: {
        handle_image_constraint(static_cast<const ImageConstraint&>(*constraint));
        break;
      }
      case Constraint::Kind::SCALE: {
        handle_scale_constraint(static_cast<const ScaleConstraint&>(*constraint));
        break;
      }
      case Constraint::Kind::BLOAT: {
        handle_bloat_constraint(static_cast<const BloatConstraint&>(*constraint));
        break;
      }
    }
  }

  // Combine states of each union of equivalence classes into one
  std::unordered_set<UnionFindEntry*> distinct_entries;

  // REVIEW: this is fishy, we are inserting the values of a map into a set, surely those
  // values are already unique? Why do we need to create another set, especially since we just
  // use this set to loop over below, why can't we just reuse the map?
  distinct_entries.reserve(table.size());
  for (auto&& [_, entry] : table) {
    distinct_entries.insert(entry);
  }

  equiv_classes_.reserve(distinct_entries.size());
  for (const auto* entry : distinct_entries) {
    auto& equiv_class = equiv_classes_.emplace_back(entry);

    for (const auto* symb : equiv_class.partition_symbols) {
      equiv_class_map_.insert({*symb, &equiv_class});
    }
  }
}

void ConstraintSolver::solve_dependent_constraints(Strategy* strategy)
{
  const auto solve_image_constraint = [&strategy](const ImageConstraint& image_constraint) {
    auto image = image_constraint.resolve(*strategy);

    strategy->insert(*image_constraint.var_range(), std::move(image));
  };

  const auto solve_scale_constraint = [&strategy](const ScaleConstraint& scale_constraint) {
    auto scaled = scale_constraint.resolve(*strategy);

    strategy->insert(*scale_constraint.var_bigger(), std::move(scaled));
  };

  const auto solve_bloat_constraint = [&strategy](const BloatConstraint& bloat_constraint) {
    auto bloated = bloat_constraint.resolve(*strategy);

    strategy->insert(*bloat_constraint.var_bloat(), std::move(bloated));
  };

  for (auto&& constraint : constraints_) {
    switch (constraint->kind()) {
      case Constraint::Kind::IMAGE: {
        solve_image_constraint(static_cast<const ImageConstraint&>(*constraint));
        break;
      }
      case Constraint::Kind::SCALE: {
        solve_scale_constraint(static_cast<const ScaleConstraint&>(*constraint));
        break;
      }
      case Constraint::Kind::BLOAT: {
        solve_bloat_constraint(static_cast<const BloatConstraint&>(*constraint));
        break;
      }
      case Constraint::Kind::ALIGNMENT: [[fallthrough]];
      case Constraint::Kind::BROADCAST: break;
    }
  }
}

Span<const Variable* const> ConstraintSolver::find_equivalence_class(
  const Variable& partition_symbol) const
{
  return equiv_class_map_.at(partition_symbol)->partition_symbols;
}

const Restrictions& ConstraintSolver::find_restrictions(const Variable& partition_symbol) const
{
  return equiv_class_map_.at(partition_symbol)->restrictions;
}

void ConstraintSolver::dump()
{
  if (!log_legate_partitioner().want_debug()) {
    return;
  }
  log_legate_partitioner().debug() << "===== Constraint Graph =====";
  log_legate_partitioner().debug() << "Stores:";
  for (auto&& symbol : partition_symbols_.elements()) {
    auto&& store = symbol->operation()->find_store(symbol);

    log_legate_partitioner().debug() << "  " << symbol->to_string() << " ~> " << store->to_string();
  }
  log_legate_partitioner().debug() << "Variables:";
  for (auto&& symbol : partition_symbols_.elements()) {
    log_legate_partitioner().debug() << "  " << symbol->to_string();
  }
  log_legate_partitioner().debug() << "Constraints:";
  for (auto&& constraint : constraints_) {
    log_legate_partitioner().debug() << "  " << constraint->to_string();
  }
  log_legate_partitioner().debug() << "============================";
}

}  // namespace legate::detail
