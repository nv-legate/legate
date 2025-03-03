/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/partitioner.h>

#include <legate/data/detail/logical_store.h>
#include <legate/operation/detail/operation.h>
#include <legate/partitioning/detail/constraint_solver.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/runtime/detail/config.h>
#include <legate/runtime/detail/region_manager.h>
#include <legate/runtime/detail/runtime.h>

#include <algorithm>
#include <tuple>

namespace legate::detail {

namespace {

class LaunchDomainResolver {
 public:
  void record_launch_domain(const Domain& launch_domain);
  void record_unbound_store(std::uint32_t unbound_dim);
  void set_must_be_sequential(bool must_be_sequential);

  [[nodiscard]] Domain resolve_launch_domain() const;

 private:
  template <typename T>
  static void set_min_optional_(const T& value, bool* on_change_flag, std::optional<T>* opt);

  bool must_be_sequential_{};
  bool must_be_1d_{};
  std::optional<std::uint32_t> unbound_dim_{};
  std::optional<Domain> launch_domain_{};
  std::optional<std::size_t> launch_volume_{};
};

// ==========================================================================================

template <typename T>
void LaunchDomainResolver::set_min_optional_(const T& value,
                                             bool* on_change_flag,
                                             std::optional<T>* opt)
{
  if (opt->has_value()) {
    if (*opt != value) {
      *on_change_flag = true;
    }
    *opt = std::min(value, **opt);
  } else {
    opt->emplace(value);
  }
}

// ==========================================================================================

void LaunchDomainResolver::record_launch_domain(const Domain& launch_domain)
{
  set_min_optional_(launch_domain, &must_be_1d_, &launch_domain_);
  set_min_optional_(launch_domain.get_volume(), &must_be_sequential_, &launch_volume_);
}

void LaunchDomainResolver::record_unbound_store(std::uint32_t unbound_dim)
{
  if (unbound_dim_.has_value() && *unbound_dim_ != unbound_dim) {
    set_must_be_sequential(true);
  } else {
    unbound_dim_ = unbound_dim;
  }
}

void LaunchDomainResolver::set_must_be_sequential(bool must_be_sequential)
{
  must_be_sequential_ = must_be_sequential;
}

Domain LaunchDomainResolver::resolve_launch_domain() const
{
  if (must_be_sequential_ || !launch_domain_.has_value()) {
    return {};
  }
  if (must_be_1d_) {
    if (unbound_dim_.value_or(0) > 1) {
      return {};
    }
    LEGATE_ASSERT(launch_volume_.has_value() && *launch_volume_ >= 1);
    return {
      0,
      static_cast<coord_t>(*launch_volume_ - 1)  // NOLINT(bugprone-unchecked-optional-access)
    };
  }

  LEGATE_ASSERT(launch_domain_.has_value());

  const auto& launch_domain = *launch_domain_;

  if (unbound_dim_.has_value() && launch_domain.dim != static_cast<int>(*unbound_dim_)) {
    return {};
  }
  return launch_domain;
}

}  // namespace

////////////////////////////////////////////////////
// legate::detail::Strategy
////////////////////////////////////////////////////

const Domain& Strategy::launch_domain(const Operation* op) const
{
  static const auto empty = Domain{};

  auto finder = launch_domains_.find(op);
  return finder != launch_domains_.end() ? finder->second : empty;
}

void Strategy::set_launch_domain(const Operation* op, const Domain& domain)
{
  LEGATE_ASSERT(launch_domains_.find(op) == launch_domains_.end());
  launch_domains_.insert({op, domain});
}

void Strategy::insert(const Variable* partition_symbol, InternalSharedPtr<Partition> partition)
{
  LEGATE_ASSERT(!has_assignment(partition_symbol));
  assignments_.insert({*partition_symbol, std::move(partition)});
}

void Strategy::insert(const Variable* partition_symbol,
                      InternalSharedPtr<Partition> partition,
                      Legion::FieldSpace field_space,
                      Legion::FieldID field_id)
{
  LEGATE_ASSERT(fields_for_unbound_stores_.find(*partition_symbol) ==
                fields_for_unbound_stores_.end());
  fields_for_unbound_stores_.insert({*partition_symbol, {field_space, field_id}});
  insert(partition_symbol, std::move(partition));
}

bool Strategy::has_assignment(const Variable* partition_symbol) const
{
  return assignments_.find(*partition_symbol) != assignments_.end();
}

const InternalSharedPtr<Partition>& Strategy::operator[](const Variable* partition_symbol) const
{
  LEGATE_ASSERT(has_assignment(partition_symbol));
  return assignments_.find(*partition_symbol)->second;
}

const std::pair<Legion::FieldSpace, Legion::FieldID>& Strategy::find_field_for_unbound_store(
  const Variable* partition_symbol) const
{
  const auto finder = fields_for_unbound_stores_.find(*partition_symbol);

  LEGATE_ASSERT(finder != fields_for_unbound_stores_.end());
  return finder->second;
}

bool Strategy::is_key_partition(const Variable* partition_symbol) const
{
  return key_partition_.has_value() && *key_partition_ == partition_symbol;
}

void Strategy::dump() const
{
  log_legate_partitioner().print() << "===== Solution =====";
  for (const auto& [symbol, part] : assignments_) {
    log_legate_partitioner().print() << symbol.to_string() << ": " << part->to_string();
  }
  for (const auto& [symbol, field] : fields_for_unbound_stores_) {
    const auto& [field_space, field_id] = field;
    log_legate_partitioner().print()
      << symbol.to_string() << ": (" << field_space << "," << field_id << ")";
  }
  for (const auto& [op, domain] : launch_domains_) {
    if (!domain.is_valid()) {
      log_legate_partitioner().print()
        << op->to_string(true /*show_provenance*/) << ": (sequential)";
    } else {
      log_legate_partitioner().print() << op->to_string(true /*show_provenance*/) << ": " << domain;
    }
  }
  log_legate_partitioner().print() << "====================";
}

void Strategy::compute_launch_domains_(const ConstraintSolver& solver)
{
  std::unordered_map<const Operation*, LaunchDomainResolver> domain_resolvers;

  for (auto&& [part_symb, partition] : assignments_) {
    const auto* op         = part_symb.operation();
    auto&& domain_resolver = domain_resolvers[op];

    if (partition->has_launch_domain()) {
      domain_resolver.record_launch_domain(partition->launch_domain());
    } else if (auto&& store = op->find_store(&part_symb); store->unbound()) {
      domain_resolver.record_unbound_store(store->dim());
    } else if (!op->supports_replicated_write() && solver.is_output(part_symb)) {
      domain_resolver.set_must_be_sequential(true);
    }
  }

  launch_domains_.reserve(domain_resolvers.size());
  for (auto&& [op, domain_resolver] : domain_resolvers) {
    launch_domains_[op] = domain_resolver.resolve_launch_domain();
  }
}

void Strategy::record_key_partition_(const Variable* partition_symbol)
{
  if (!key_partition_.has_value()) {
    key_partition_ = partition_symbol;
  }
}

////////////////////////////////////////////////////
// legate::detail::Partitioner
////////////////////////////////////////////////////

std::unique_ptr<Strategy> Partitioner::partition_stores()
{
  ConstraintSolver solver;

  for (auto&& op : operations_) {
    op->add_to_solver(solver);
  }

  solver.solve_constraints();

  if (Config::get_config().log_partitioning_decisions()) {
    solver.dump();
  }

  auto strategy = std::make_unique<Strategy>();

  // Copy the list of partition symbols as we will sort them inplace
  auto remaining_symbols =
    handle_unbound_stores_(strategy.get(), solver.partition_symbols(), solver);

  auto comparison_key = [&solver](const auto& part_symb) {
    auto* op   = part_symb->operation();
    auto store = op->find_store(part_symb);
    auto has_key_part =
      store->has_key_partition(op->machine(), solver.find_restrictions(part_symb));

    LEGATE_ASSERT(!store->unbound());
    return std::make_tuple(
      store->storage_size(), has_key_part, solver.find_access_mode(*part_symb));
  };

  std::stable_sort(remaining_symbols.begin(),
                   remaining_symbols.end(),
                   [&comparison_key](const auto& part_symb_a, const auto& part_symb_b) {
                     return comparison_key(part_symb_a) > comparison_key(part_symb_b);
                   });

  for (auto* part_symb : remaining_symbols) {
    if (strategy->has_assignment(part_symb) || solver.is_dependent(*part_symb)) {
      continue;
    }

    const auto& equiv_class  = solver.find_equivalence_class(part_symb);
    const auto& restrictions = solver.find_restrictions(part_symb);

    auto* op = part_symb->operation();
    auto partition =
      op->find_store(part_symb)->find_or_create_key_partition(op->machine(), restrictions);

    strategy->record_key_partition_(part_symb);
    LEGATE_ASSERT(partition != nullptr);
    for (auto symb : equiv_class) {
      strategy->insert(symb, partition);
    }
  }

  solver.solve_dependent_constraints(*strategy);

  strategy->compute_launch_domains_(solver);

  if (Config::get_config().log_partitioning_decisions()) {
    strategy->dump();
  }

  return strategy;
}

std::vector<const Variable*> Partitioner::handle_unbound_stores_(
  Strategy* strategy,
  std::vector<const Variable*> partition_symbols,
  const ConstraintSolver& solver)
{
  const auto runtime    = Runtime::get_runtime();
  auto is_unbound_store = [&](const Variable* part_symb) {
    if (strategy->has_assignment(part_symb)) {
      return true;
    }

    if (!part_symb->operation()->find_store(part_symb)->unbound()) {
      return false;
    }

    auto&& equiv_class = solver.find_equivalence_class(part_symb);
    const InternalSharedPtr<Partition> partition{create_no_partition()};
    auto field_space   = runtime->create_field_space();
    auto next_field_id = RegionManager::FIELD_ID_BASE;

    for (auto* symb : equiv_class) {
      if (next_field_id - RegionManager::FIELD_ID_BASE >= RegionManager::MAX_NUM_FIELDS) {
        field_space   = runtime->create_field_space();
        next_field_id = RegionManager::FIELD_ID_BASE;
      }
      auto field_id =
        runtime->allocate_field(field_space, next_field_id++, symb->store()->type()->size());
      strategy->insert(symb, partition, field_space, field_id);
    }
    return true;
  };

  partition_symbols.erase(
    std::remove_if(partition_symbols.begin(), partition_symbols.end(), std::move(is_unbound_store)),
    partition_symbols.end());
  return partition_symbols;
}

}  // namespace legate::detail
