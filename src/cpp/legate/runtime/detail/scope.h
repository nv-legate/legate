/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

class Scope {
  using Machine = legate::mapping::detail::Machine;

 public:
  [[nodiscard]] std::int32_t priority() const;
  [[nodiscard]] ExceptionMode exception_mode() const;
  [[nodiscard]] ZStringView provenance() const;
  [[nodiscard]] const InternalSharedPtr<Machine>& machine() const;
  /*
   * @return The parallel_policy of this Scope.
   */
  [[nodiscard]] const ParallelPolicy& parallel_policy() const;

  [[nodiscard]] std::int32_t exchange_priority(std::int32_t priority);
  [[nodiscard]] ExceptionMode exchange_exception_mode(ExceptionMode exception_mode);
  [[nodiscard]] std::string exchange_provenance(std::string provenance);
  [[nodiscard]] InternalSharedPtr<Machine> exchange_machine(InternalSharedPtr<Machine> machine);
  /*
   * @brief Exchange the new parallel_policy with the old one.
   *
   * If the policies are different then we exchange it and flush the
   * current scheduling window.
   *
   * @return The old parallel_policy of this Scope.
   */
  [[nodiscard]] ParallelPolicy exchange_parallel_policy(ParallelPolicy parallel_policy);

 private:
  std::int32_t priority_{static_cast<std::int32_t>(TaskPriority::DEFAULT)};
  ExceptionMode exception_mode_{ExceptionMode::IMMEDIATE};
  std::string provenance_{};
  InternalSharedPtr<Machine> machine_{};
  ParallelPolicy parallel_policy_{};
};

}  // namespace legate::detail

#include <legate/runtime/detail/scope.inl>
