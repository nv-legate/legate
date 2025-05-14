/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/runtime/detail/scope.h>

#include <utility>

namespace legate::detail {

inline std::int32_t Scope::priority() const { return priority_; }

inline ExceptionMode Scope::exception_mode() const { return exception_mode_; }

inline ZStringView Scope::provenance() const { return provenance_; }

inline const InternalSharedPtr<Scope::Machine>& Scope::machine() const { return machine_; }

inline const ParallelPolicy& Scope::parallel_policy() const { return parallel_policy_; }

inline std::int32_t Scope::exchange_priority(std::int32_t priority)
{
  return std::exchange(priority_, priority);
}

inline ExceptionMode Scope::exchange_exception_mode(ExceptionMode exception_mode)
{
  return std::exchange(exception_mode_, exception_mode);
}

inline std::string Scope::exchange_provenance(std::string provenance)
{
  return std::exchange(provenance_, std::move(provenance));
}

inline InternalSharedPtr<Scope::Machine> Scope::exchange_machine(InternalSharedPtr<Machine> machine)
{
  return std::exchange(machine_, std::move(machine));
}

}  // namespace legate::detail
