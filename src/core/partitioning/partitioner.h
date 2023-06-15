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

#include <memory>
#include <optional>
#include <unordered_map>

#include "core/data/shape.h"
#include "core/partitioning/constraint.h"
#include "core/partitioning/restriction.h"
#include "legion.h"

namespace legate {
class Operation;
class Partition;
}  // namespace legate

namespace legate::detail {

class ConstraintSolver;
class LogicalStore;
class Partitioner;
class ProjectionInfo;

class Strategy {
  friend class Partitioner;

 public:
  Strategy();

 public:
  bool parallel(const Operation* op) const;
  const Domain* launch_domain(const Operation* op) const;
  void set_launch_shape(const Operation* op, const Shape& shape);

 public:
  void insert(const Variable* partition_symbol, std::shared_ptr<Partition> partition);
  void insert(const Variable* partition_symbol,
              std::shared_ptr<Partition> partition,
              Legion::FieldSpace field_space);
  bool has_assignment(const Variable* partition_symbol) const;
  std::shared_ptr<Partition> operator[](const Variable* partition_symbol) const;
  const Legion::FieldSpace& find_field_space(const Variable* partition_symbol) const;
  bool is_key_partition(const Variable* partition_symbol) const;

 public:
  void dump() const;

 private:
  void compute_launch_domains(const ConstraintSolver& solver);
  void record_key_partition(const Variable* partition_symbol);

 private:
  std::map<const Variable, std::shared_ptr<Partition>> assignments_{};
  std::map<const Variable, Legion::FieldSpace> field_spaces_{};
  std::map<const Operation*, std::unique_ptr<Domain>> launch_domains_{};
  std::optional<const Variable*> key_partition_{};
};

class Partitioner {
 public:
  Partitioner(std::vector<Operation*>&& operations);

 public:
  std::unique_ptr<Strategy> partition_stores();

 private:
  // Populates solutions for unbound stores in the `strategy` and returns remaining partition
  // symbols
  std::vector<const Variable*> handle_unbound_stores(
    Strategy* strategy,
    const std::vector<const Variable*>& partition_symbols,
    const ConstraintSolver& constraints);

 private:
  std::vector<Operation*> operations_;
};

}  // namespace legate::detail
