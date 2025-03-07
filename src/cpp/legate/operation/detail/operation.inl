/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/detail/operation.h>

namespace legate::detail {

inline bool Operation::StoreArg::needs_flush() const { return store->needs_flush(); }

// ==========================================================================================

inline void Operation::validate() {}

inline void Operation::add_to_solver(ConstraintSolver& /*solver*/) {}

inline void Operation::launch() { LEGATE_ABORT("This method should have been overridden"); }

inline void Operation::launch(Strategy* /*strategy*/)
{
  LEGATE_ABORT("This method should have been overridden");
}

inline bool Operation::supports_replicated_write() const { return false; }

inline std::int32_t Operation::priority() const { return priority_; }

inline const mapping::detail::Machine& Operation::machine() const { return machine_; }

inline ZStringView Operation::provenance() const { return provenance_; }

}  // namespace legate::detail
