/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/detail/constraint.h>
#include <legate/utilities/hash.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/typedefs.h>

#include <optional>
#include <unordered_map>
#include <utility>

namespace legate::detail {

class Operation;
class Partition;
class ConstraintSolver;

class Strategy {
 public:
  Strategy() = default;

  // NOTE(amberhassaan): disabled only because copy is expensive.
  Strategy(const Strategy&)            = delete;
  Strategy& operator=(const Strategy&) = delete;

  Strategy(Strategy&&) noexcept            = default;
  Strategy& operator=(Strategy&&) noexcept = default;

  [[nodiscard]] bool parallel(const Operation& op) const;
  [[nodiscard]] const Domain& launch_domain(const Operation& op) const;
  void set_launch_domain(const Operation& op, const Domain& domain);

  void insert(const Variable& partition_symbol, InternalSharedPtr<Partition> partition);
  void insert(const Variable& partition_symbol,
              InternalSharedPtr<Partition> partition,
              Legion::FieldSpace field_space,
              Legion::FieldID field_id);
  [[nodiscard]] bool has_assignment(const Variable& partition_symbol) const;
  [[nodiscard]] const InternalSharedPtr<Partition>& operator[](
    const Variable& partition_symbol) const;
  [[nodiscard]] const std::pair<Legion::FieldSpace, Legion::FieldID>& find_field_for_unbound_store(
    const Variable& partition_symbol) const;
  [[nodiscard]] bool is_key_partition(const Variable& partition_symbol) const;

  void dump() const;

  class PrivateKey {
    friend class Strategy;
    friend class Partitioner;
  };

  void compute_launch_domains(PrivateKey, const ConstraintSolver& solver);
  void record_key_partition(PrivateKey, const Variable& partition_symbol);

 private:
  std::unordered_map<Variable, InternalSharedPtr<Partition>> assignments_{};
  std::unordered_map<Variable, std::pair<Legion::FieldSpace, Legion::FieldID>>
    fields_for_unbound_stores_{};
  std::unordered_map<const Operation*, Domain> launch_domains_{};
  std::optional<const Variable*> key_partition_{};
};

}  // namespace legate::detail
