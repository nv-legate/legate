/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/detail/access_mode.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/runtime/detail/streaming/base_operation_checker.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <optional>
#include <string_view>
#include <unordered_map>

namespace legate::detail {

class Operation;
class Strategy;
class StreamingErrorContext;
class LogicalStore;

/**
 * @brief Checks if the accesses of an Operation are valid for streaming execution
 * mode.
 *
 * For streaming mode execution, the following conditions suffice validity for
 * streaming execution:
 * 1. Accesses to the stores must be point-wise, i.e.,
 *    an operation must depend on at most one partition of each store it accesses.
 * 2. A store must not be partitioned in more than one way across operations inside
 *    a streaming scope.
 * 3. Reductions must be the last operation on a store, i.e., a later operation
 *    within the streaming scope cannot read or write to the store being reduced.
 */
class DependencyChecker final : public BaseOperationChecker {
 public:
  /**
   * @return true if the operation's data dependencies are valid for streaming
   * execution.
   *
   * @see BaseOperationChecker::is_streamable
   */
  [[nodiscard]] bool is_streamable(const InternalSharedPtr<Operation>& op,
                                   const std::optional<InternalSharedPtr<Strategy>>& maybe_strategy,
                                   StreamingErrorContext* ctx) override;
  /**
   * @brief Returns name for printing errors.
   */
  [[nodiscard]] std::string_view name() const override;

 private:
  /**
   * @brief Bookkeeping info about store accesses.
   */
  struct AccessInfo {
    InternalSharedPtr<Operation> op;
    InternalSharedPtr<Partition> partition;
    AccessMode access_mode;
  };

  /**
   * @brief Compare partitions of two accesses to the same root storage.
   *
   * @param store Store pointer.
   * @param last Last access to the storage.
   * @param cur Current access to the storage.
   *
   * @return true if both accesses use the same partitioning of the storage.
   */
  [[nodiscard]] bool have_equal_partitioning_(const InternalSharedPtr<LogicalStore>& store,
                                              const AccessInfo& last,
                                              const AccessInfo& cur);

  /**
   * @brief Get the last access recorded for the store.
   *
   * Note: Accesses are grouped by root storage of the store.
   *
   * @param store The store.
   *
   * @return Reference to last access if there was a previous task accessing this
   * store.
   */
  [[nodiscard]] std::optional<AccessInfo>& get_last_access_(
    const InternalSharedPtr<LogicalStore>& store);

  /**
   * @brief Analyze dependencies of all the input stores of an operation.
   *
   * @param op The operation.
   * @param strategy The strategy.
   * @param ctx Context collector for user readable error messages.
   *
   * @return true if the input dependencies are streamable, false otherwise.
   */
  [[nodiscard]] bool analyze_inputs_(const InternalSharedPtr<Operation>& op,
                                     const Strategy& strategy,
                                     StreamingErrorContext* ctx);

  /**
   * @brief Analyze dependencies of all the output stores of an operation.
   *
   * @param op The operation.
   * @param strategy The strategy.
   * @param ctx Context collector for user readable error messages.
   *
   * @return true if the output dependencies are streamable, false otherwise.
   */
  [[nodiscard]] bool analyze_outputs_(const InternalSharedPtr<Operation>& op,
                                      const Strategy& strategy,
                                      StreamingErrorContext* ctx);

  /**
   * @brief Analyze dependencies of all the reduced stores of an operation.
   *
   * @param op The operation.
   * @param strategy The strategy.
   * @param ctx Context collector for user readable error messages.
   *
   * @return true if the reduction dependencies are streamable, false otherwise.
   */
  [[nodiscard]] bool analyze_reductions_(const InternalSharedPtr<Operation>& op,
                                         const Strategy& strategy,
                                         StreamingErrorContext* ctx);

  /**
   * @brief Helper function that composes a user readable error message.
   *
   * @param store The store being analyzed.
   * @param curr Current access info.
   * @param last Last access info.
   * @param reason Reason for why dependency was non-streamable.
   * @param ctx Context collector for user readable error messages.
   *
   * @return false, always, because dependencies were non-streamable.
   */
  static bool fail_with_msg_(const InternalSharedPtr<LogicalStore>& store,
                             const AccessInfo& curr,
                             const AccessInfo& last,
                             std::string_view reason,
                             StreamingErrorContext* ctx);

  // key: root storage ID.
  // value: most recent access to the store by an operation (in program order).
  std::unordered_map<std::uint64_t, std::optional<AccessInfo>> per_store_accesses_{};
};

}  // namespace legate::detail
