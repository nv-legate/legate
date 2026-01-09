/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/strategy.h>

#include <legate/operation/detail/operation.h>
#include <legate/partitioning/detail/constraint_solver.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/detail/formatters.h>

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>

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
    LEGATE_ASSERT(launch_volume_ >= 1);
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

const Domain& Strategy::launch_domain(const Operation& op) const
{
  static const auto empty = Domain{};
  const auto it           = launch_domains_.find(&op);

  return it != launch_domains_.end() ? it->second : empty;
}

void Strategy::set_launch_domain(const Operation& op, const Domain& domain)
{
  LEGATE_ASSERT(launch_domains_.find(&op) == launch_domains_.end());
  launch_domains_.insert({&op, domain});
}

void Strategy::insert(const Variable& partition_symbol, InternalSharedPtr<Partition> partition)
{
  LEGATE_ASSERT(!has_assignment(partition_symbol));
  assignments_.insert({partition_symbol, std::move(partition)});
}

void Strategy::insert(const Variable& partition_symbol,
                      InternalSharedPtr<Partition> partition,
                      Legion::FieldSpace field_space,
                      Legion::FieldID field_id)
{
  LEGATE_ASSERT(fields_for_unbound_stores_.find(partition_symbol) ==
                fields_for_unbound_stores_.end());
  fields_for_unbound_stores_.insert({partition_symbol, {field_space, field_id}});
  insert(partition_symbol, std::move(partition));
}

bool Strategy::has_assignment(const Variable& partition_symbol) const
{
  return assignments_.find(partition_symbol) != assignments_.end();
}

const InternalSharedPtr<Partition>& Strategy::operator[](const Variable& partition_symbol) const
{
  const auto it = assignments_.find(partition_symbol);

  LEGATE_ASSERT(it != assignments_.end());
  return it->second;
}

const std::pair<Legion::FieldSpace, Legion::FieldID>& Strategy::find_field_for_unbound_store(
  const Variable& partition_symbol) const
{
  const auto finder = fields_for_unbound_stores_.find(partition_symbol);

  LEGATE_ASSERT(finder != fields_for_unbound_stores_.end());
  return finder->second;
}

bool Strategy::is_key_partition(const Variable& partition_symbol) const
{
  return key_partition_ == &partition_symbol;
}

void Strategy::dump() const
{
  if (!log_legate_partitioner().want_debug()) {
    return;
  }
  log_legate_partitioner().debug() << "===== Solution =====";
  for (const auto& [symbol, part] : assignments_) {
    log_legate_partitioner().debug() << symbol.to_string() << ": " << part->to_string();
  }
  for (const auto& [symbol, field] : fields_for_unbound_stores_) {
    const auto& [field_space, field_id] = field;

    log_legate_partitioner().debug()
      << symbol.to_string() << ": (" << field_space << "," << field_id << ")";
  }
  for (const auto& [op, domain] : launch_domains_) {
    if (!domain.is_valid()) {
      log_legate_partitioner().debug()
        << op->to_string(true /*show_provenance*/) << ": (sequential)";
    } else {
      log_legate_partitioner().debug() << op->to_string(true /*show_provenance*/) << ": " << domain;
    }
  }
  log_legate_partitioner().debug() << "====================";
}

void Strategy::compute_launch_domains(PrivateKey, const ConstraintSolver& solver)
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

void Strategy::record_key_partition(PrivateKey, const Variable& partition_symbol)
{
  if (!key_partition_.has_value()) {
    key_partition_ = &partition_symbol;
  }
}

bool Strategy::parallel(const Operation& op) const { return launch_domain(op).is_valid(); }

}  // namespace legate::detail
