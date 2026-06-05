/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/strategy.h>

#include <legate/operation/detail/operation.h>
#include <legate/partitioning/detail/constraint_solver.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/partitioning/detail/partition/no_partition.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/detail/linearize.h>

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>

namespace legate::detail {

DomainPoint Strategy::task_index() const
{
  if (!launch_domain().is_valid()) {
    return {};
  }

  const auto node_id = Runtime::get_runtime().node_id();

  return delinearize(launch_domain().lo(), launch_domain().hi(), node_id);
}

void Strategy::insert(const Variable& partition_symbol, InternalSharedPtr<Partition> partition)
{
  LEGATE_ASSERT(!has_assignment(partition_symbol));
  assignments_.insert({partition_symbol, std::move(partition)});
}

bool Strategy::has_assignment(const Variable& partition_symbol) const
{
  return assignments_.find(partition_symbol) != assignments_.end();
}

const InternalSharedPtr<Partition>& Strategy::operator[](const Variable& partition_symbol) const
{
  const auto it = assignments_.find(partition_symbol);

  if (it == assignments_.end()) {
    return create_no_partition();
  }

  return it->second;
}

void Strategy::dump() const
{
  if (!log_legate_partitioner().want_debug()) {
    return;
  }
  log_legate_partitioner().debug() << "===== Solution =====";
  if (!launch_domain_.is_valid()) {
    log_legate_partitioner().debug()
      << operation_->to_string(true /*show_provenance*/) << ": (sequential)";
  } else {
    log_legate_partitioner().debug()
      << operation_->to_string(true /*show_provenance*/) << ": " << launch_domain_;
  }
  for (const auto& [symbol, part] : assignments_) {
    log_legate_partitioner().debug() << symbol.to_string() << ": " << part->to_string();
  }
  log_legate_partitioner().debug() << "====================";
}

void Strategy::record_key_partition(PrivateKey, const Variable& partition_symbol)
{
  if (!key_partition_.has_value()) {
    key_partition_ = partition_symbol;
  }
}

const SymbolicPoint& Strategy::find_store_projection(const Variable& partition_symbol) const
{
  auto finder = projections_.find(partition_symbol);

  if (finder == projections_.end()) {
    static const auto empty_projection = SymbolicPoint{};

    return empty_projection;
  }

  return finder->second;
}

const SymbolicPoint& Strategy::find_key_store_projection() const
{
  if (!key_partition_.has_value()) {
    static const auto empty_projection = SymbolicPoint{};

    // If no key partition exists, use the identity function (of ID 0)
    return empty_projection;
  }

  return find_store_projection(*key_partition_);
}

}  // namespace legate::detail
