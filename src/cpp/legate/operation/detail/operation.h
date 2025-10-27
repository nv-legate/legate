/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/logical_store.h>
#include <legate/data/detail/logical_store_partition.h>
#include <legate/mapping/detail/machine.h>
#include <legate/operation/detail/access_mode.h>
#include <legate/operation/detail/store_projection.h>
#include <legate/partitioning/detail/constraint.h>
#include <legate/tuning/parallel_policy.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/detail/hash.h>
#include <legate/utilities/detail/zstring_view.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <fmt/format.h>

#include <deque>
#include <string>
#include <unordered_map>

namespace legate::detail {

class ConstraintSolver;
class Strategy;

enum class AccessMode : std::uint8_t;

class Operation {
 protected:
  /**
   * @brief Construct an Operation.
   * @param unique_id Runtime assigned dynamic unique id for each instance.
   */
  explicit Operation(std::uint64_t unique_id);

  /**
   * @brief Construct an Operation.
   * @param unique_id Runtime assigned dynamic unique id for each instance.
   * @param priority Scheduling Priority. @see legate::Scope
   * @machine Machine subset to schedule the operation on.
   */
  Operation(std::uint64_t unique_id, std::int32_t priority, mapping::detail::Machine machine);

 public:
  /**
   * @brief
   * Class to pair stores and their associated variable symbols.
   */
  class StoreArg {
   public:
    /**
     * @return true if store needs flush.
     */
    [[nodiscard]] bool needs_flush() const;
    InternalSharedPtr<LogicalStore> store{};
    const Variable* variable{};
  };

  /**
   * @brief. Operation kind assigned for each subclass.
   */
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
    PHYSICAL_TASK,
    REDUCE,
    RELEASE_REGION_FIELD,
    SCATTER,
    SCATTER_GATHER,
    TIMING,
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
  [[nodiscard]] virtual std::string to_string(bool show_provenance) const;

  /**
   * @brief Whether the operation should immediately flush the scheduling window (regardless of
   * size) when it is submitted.
   *
   * @return `true` if the scheduling window should be flushed, `false` otherwise.
   *
   * A potential reason for needing an immediate flush is that at least one argument to the
   * operation is unbound. In this case, the operation should be scheduled as soon as possible
   * to allow downstream tasks to block on its completion to learn the true size of the store.
   */
  [[nodiscard]] virtual bool needs_flush() const = 0;

  [[nodiscard]] virtual bool supports_replicated_write() const;
  /**
   * When an operation supports streaming it is treated specially inside a Scope that is Streaming.
   * We track if this operation is the last user of Region Requirement inside a Streaming Scope
   * and if it is we then assign a Discard Flag to that Region Requirement marking the Region
   * as collectable after this operation is complete.
   *
   * @return Whether the operation supports streaming.
   */
  [[nodiscard]] virtual bool supports_streaming() const;

  /**
   * @brief Whether the operation requires automatic partitioning analysis before launch.
   *
   * Operations that require partitioning analysis are launched via the `launch(Strategy*)`
   * overload. Operations that do NOT require the analysis, are launched via the `launch()`
   * overload.
   *
   * Operations may not require partitioning analysis for a number of reasons:
   *
   * 1. The operation does its own bespoke (or user-specified) partitioning. For example,
   *    `ManualTask` has its partitioning specified by the user.
   * 2. The operation is partition-agnostic. For example, `ExecutionFence` does not require any
   *    partitioning in order to run.
   *
   * @return `true` if the operation requires partitioning analysis to be performed, `false`
   * otherwise.
   */
  [[nodiscard]] virtual bool needs_partitioning() const = 0;

  [[nodiscard]] const Variable* find_or_declare_partition(
    const InternalSharedPtr<LogicalStore>& store);
  [[nodiscard]] const Variable* declare_partition();
  [[nodiscard]] const InternalSharedPtr<LogicalStore>& find_store(const Variable* variable) const;

  [[nodiscard]] std::int32_t priority() const;
  [[nodiscard]] const mapping::detail::Machine& machine() const;
  /*
   * @return The parallel_policy of this operation.
   */
  [[nodiscard]] const ParallelPolicy& parallel_policy() const;
  [[nodiscard]] ZStringView provenance() const;

  /**
   * @brief Return all the stores read by this operation.
   *
   * @return a vector of input StoreArg objects.
   */
  [[nodiscard]] const SmallVector<StoreArg>& input_stores() const;
  /**
   * @brief Return all the stores written by this operation.
   *
   * @return a vector of output StoreArg objects.
   */
  [[nodiscard]] const SmallVector<StoreArg>& output_stores() const;
  /**
   * @brief Return all the stores reduced by this operation.
   *
   * @return a vector of StoreArg objects used in reductions.
   */
  [[nodiscard]] const SmallVector<StoreArg>& reduction_stores() const;

 protected:
  /**
   * @brief Register a store access with the base class.
   *
   * Derived classes must call this method for every store that they access so that
   * the dependencies can be tracked accurately.
   *
   * @param variable pointer to the symbol created for the store being accessed.
   *
   * @param store pointer to the store being accessed.
   *
   * @param access_mode the mode of access.
   */
  void record_partition_(const Variable* variable,
                         InternalSharedPtr<LogicalStore> store,
                         AccessMode access_mode);

  // Helper methods
  [[nodiscard]] static StoreProjection create_store_projection_(const Strategy& strategy,
                                                                const Domain& launch_domain,
                                                                const StoreArg& arg);

  std::uint64_t unique_id_{};
  std::int32_t next_part_id_{};
  std::int32_t priority_{static_cast<std::int32_t>(TaskPriority::DEFAULT)};
  std::deque<Variable> partition_symbols_{};
  std::unordered_map<std::reference_wrapper<const Variable>, InternalSharedPtr<LogicalStore>>
    store_mappings_{};
  std::unordered_map<InternalSharedPtr<LogicalStore>, const Variable*> part_mappings_{};

  // Derived classes populate these vectors via calls to record_partition_
  SmallVector<StoreArg> input_args_{};
  SmallVector<StoreArg> output_args_{};
  SmallVector<StoreArg> reduction_args_{};

  std::string provenance_{};
  mapping::detail::Machine machine_{};
  ParallelPolicy parallel_policy_{};
};

}  // namespace legate::detail

#include <legate/operation/detail/operation.inl>

namespace fmt {

template <>
struct formatter<legate::detail::Operation::Kind> : formatter<legate::detail::ZStringView> {
  format_context::iterator format(legate::detail::Operation::Kind kind, format_context& ctx) const;
};

}  // namespace fmt
