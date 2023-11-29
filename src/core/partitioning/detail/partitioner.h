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

#pragma once

#include "core/data/shape.h"
#include "core/partitioning/detail/constraint.h"

#include <map>
#include <memory>
#include <optional>
#include <vector>

namespace legate {
struct Partition;
}  // namespace legate

namespace legate::detail {

struct ConstraintSolver;
class LogicalStore;
class Partitioner;

class Strategy {
  friend class Partitioner;

 public:
  [[nodiscard]] bool parallel(const Operation* op) const;
  [[nodiscard]] Domain launch_domain(const Operation* op) const;
  void set_launch_domain(const Operation* op, const Domain& domain);

  void insert(const Variable* partition_symbol, std::shared_ptr<Partition> partition);
  void insert(const Variable* partition_symbol,
              std::shared_ptr<Partition> partition,
              Legion::FieldSpace field_space);
  [[nodiscard]] bool has_assignment(const Variable* partition_symbol) const;
  [[nodiscard]] std::shared_ptr<Partition> operator[](const Variable* partition_symbol) const;
  [[nodiscard]] const Legion::FieldSpace& find_field_space(const Variable* partition_symbol) const;
  [[nodiscard]] bool is_key_partition(const Variable* partition_symbol) const;

  void dump() const;

 private:
  void compute_launch_domains(const ConstraintSolver& solver);
  void record_key_partition(const Variable* partition_symbol);

  std::map<const Variable, std::shared_ptr<Partition>> assignments_{};
  std::map<const Variable, Legion::FieldSpace> field_spaces_{};
  std::map<const Operation*, Domain> launch_domains_{};
  std::optional<const Variable*> key_partition_{};
};

class Partitioner {
 public:
  Partitioner(std::vector<Operation*>&& operations);

  [[nodiscard]] std::unique_ptr<Strategy> partition_stores();

 private:
  // Populates solutions for unbound stores in the `strategy` and returns remaining partition
  // symbols
  [[nodiscard]] static std::vector<const Variable*> handle_unbound_stores(
    Strategy* strategy,
    const std::vector<const Variable*>& partition_symbols,
    const ConstraintSolver& solver);

  std::vector<Operation*> operations_{};
};

}  // namespace legate::detail

#include "core/partitioning/detail/partitioner.inl"
