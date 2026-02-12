/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/mapping/detail/machine.h>
#include <legate/runtime/exception_mode.h>
#include <legate/tuning/parallel_policy.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/detail/zstring_view.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <cstdint>
#include <string>

namespace legate::detail {

class Config;

class Scope {
  using Machine = legate::mapping::detail::Machine;

 public:
  /**
   * @brief Enum type that signals the nature of the scope change.
   */
  enum class ChangeKind : std::uint8_t {
    /**
     * @brief New scope has started.
     */
    SCOPE_BEG,
    /**
     * @brief Existing scope has ended.
     */
    SCOPE_END
  };

  [[nodiscard]] std::int32_t priority() const;
  [[nodiscard]] ExceptionMode exception_mode() const;
  [[nodiscard]] ZStringView provenance() const;
  [[nodiscard]] const InternalSharedPtr<Machine>& machine() const;
  /*
   * @return The parallel_policy of this Scope.
   */
  [[nodiscard]] const ParallelPolicy& parallel_policy() const;

  /**
   * @return The scheduling window size of this Scope.
   */
  [[nodiscard]] std::uint32_t scheduling_window_size() const;

  [[nodiscard]] std::int32_t exchange_priority(std::int32_t priority);
  [[nodiscard]] ExceptionMode exchange_exception_mode(ExceptionMode exception_mode);
  [[nodiscard]] std::string exchange_provenance(std::string provenance);
  [[nodiscard]] InternalSharedPtr<Machine> exchange_machine(InternalSharedPtr<Machine> machine);
  /*
   * @brief Exchange the new parallel_policy with the old one.
   *
   * If the policies are different then we exchange it and flush the current scheduling
   * window. Additionally, if either current or previous parallel policy is streaming, a
   * mapping fence is issued.
   *
   * @param new_policy The parallel policy.
   *
   * @return The old parallel_policy of this Scope.
   */
  [[nodiscard]] ParallelPolicy exchange_parallel_policy(ParallelPolicy new_policy);

  /**
   * @brief Initiate actions required when ParallelPolicy is about to be replaced.
   *
   * When ParallelPolicy is replaced, it may trigger additional actions, such as
   * flushing the scheduling window. This function separates those actions from the
   * actual exchange of the policy because the said actions may throw exceptions,
   * and if that happens, we still want to exchange the ParallelPolicy after all.
   *
   * @param new_policy New ParallelPolicy.
   * @param ChangeKind Denotes beginning or ending of a scope.
   *
   * @throws std::invalid_argument exception if operations submitted in the scope
   * are not streamable.
   */
  void trigger_exchange_side_effects(const ParallelPolicy& new_policy,
                                     ChangeKind change_kind) const;

  /**
   * @brief Exchange the new scheduling window size with the old one.
   *
   * @param window_size The new window size.
   *
   * @return The old window size.
   */
  [[nodiscard]] std::uint32_t exchange_scheduling_window_size(std::uint32_t window_size);

  /**
   * @brief Initialize global scope.
   *
   * @param config Legate configuration.
   */
  explicit Scope(const Config& config);

 private:
  std::int32_t priority_{static_cast<std::int32_t>(TaskPriority::DEFAULT)};
  ExceptionMode exception_mode_{ExceptionMode::IMMEDIATE};
  std::string provenance_{};
  InternalSharedPtr<Machine> machine_{};
  std::uint32_t scheduling_window_size_{1};

  ParallelPolicy parallel_policy_;
};

}  // namespace legate::detail

#include <legate/runtime/detail/scope.inl>
