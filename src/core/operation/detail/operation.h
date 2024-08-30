/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "core/data/detail/logical_store.h"
#include "core/mapping/detail/machine.h"
#include "core/operation/detail/store_projection.h"
#include "core/partitioning/detail/constraint.h"
#include "core/utilities/detail/core_ids.h"
#include "core/utilities/detail/formatters.h"
#include "core/utilities/detail/hash.h"
#include "core/utilities/detail/zstring_view.h"
#include "core/utilities/internal_shared_ptr.h"

#include <deque>
#include <fmt/format.h>
#include <memory>
#include <string>
#include <unordered_map>

namespace legate::detail {
class ConstraintSolver;
class Strategy;

class Operation {
 protected:
  class StoreArg {
   public:
    [[nodiscard]] bool needs_flush() const;
    InternalSharedPtr<LogicalStore> store{};
    const Variable* variable{};
  };

  explicit Operation(std::uint64_t unique_id);
  Operation(std::uint64_t unique_id, std::int32_t priority, mapping::detail::Machine machine);

 public:
  enum class Kind : std::uint8_t {
    ATTACH,
    AUTO_TASK,
    COPY,
    DISCARD,
    EXECUTION_FENCE,
    FILL,
    GATHER,
    INDEX_ATTACH,
    MANUAL_TASK,
    MAPPING_FENCE,
    REDUCE,
    SCATTER,
    SCATTER_GATHER,
    TIMING,
    UNMAP_AND_DETACH,
  };

  virtual ~Operation() = default;

  // These methods do nothing by default. Concrete operation types can optionally implement logic
  // specific to them
  virtual void validate();
  virtual void add_to_solver(ConstraintSolver& solver);
  // Though these aren't pure virtual methods, concrete operation types must implement at least one
  // of them and also add themselves to the switch statement in `needs_partitioning()` such that the
  // function would return `true`/`false` if the operation implemented the
  // `launch(Strategy*)`/`launch()` override.
  virtual void launch();
  virtual void launch(Strategy* strategy);

  [[nodiscard]] virtual Kind kind() const = 0;
  [[nodiscard]] virtual std::string to_string() const;
  [[nodiscard]] virtual bool needs_flush() const;
  [[nodiscard]] virtual bool supports_replicated_write() const;
  // When `is_internal()` returns `true` on an operation, the runtime skips validation and the flush
  // check on the operation.
  [[nodiscard]] bool is_internal() const;
  [[nodiscard]] bool needs_partitioning() const;

  [[nodiscard]] const Variable* find_or_declare_partition(
    const InternalSharedPtr<LogicalStore>& store);
  [[nodiscard]] const Variable* declare_partition();
  [[nodiscard]] const InternalSharedPtr<LogicalStore>& find_store(const Variable* variable) const;

  [[nodiscard]] std::int32_t priority() const;
  [[nodiscard]] const mapping::detail::Machine& machine() const;
  [[nodiscard]] ZStringView provenance() const;

 protected:
  void record_partition_(const Variable* variable, InternalSharedPtr<LogicalStore> store);
  // Helper methods
  [[nodiscard]] static std::unique_ptr<StoreProjection> create_store_projection_(
    const Strategy& strategy, const Domain& launch_domain, const StoreArg& arg);

  std::uint64_t unique_id_{};
  std::int32_t next_part_id_{};
  std::int32_t priority_{static_cast<std::int32_t>(TaskPriority::DEFAULT)};
  std::deque<Variable> partition_symbols_{};
  std::unordered_map<std::reference_wrapper<const Variable>, InternalSharedPtr<LogicalStore>>
    store_mappings_{};
  std::unordered_map<InternalSharedPtr<LogicalStore>, const Variable*> part_mappings_{};
  std::string provenance_{};
  mapping::detail::Machine machine_{};
};

}  // namespace legate::detail

#include "core/operation/detail/operation.inl"

namespace fmt {

template <>
struct formatter<legate::detail::Operation::Kind> : formatter<legate::detail::ZStringView> {
  format_context::iterator format(legate::detail::Operation::Kind kind, format_context& ctx) const;
};

}  // namespace fmt
