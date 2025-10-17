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

inline bool Operation::supports_replicated_write() const { return false; }

inline bool Operation::supports_streaming() const { return needs_partitioning(); }

inline std::int32_t Operation::priority() const { return priority_; }

inline const mapping::detail::Machine& Operation::machine() const { return machine_; }

inline const ParallelPolicy& Operation::parallel_policy() const { return parallel_policy_; }

inline ZStringView Operation::provenance() const { return provenance_; }

inline const SmallVector<Operation::StoreArg>& Operation::input_stores() const
{
  return input_args_;
}

inline const SmallVector<Operation::StoreArg>& Operation::output_stores() const
{
  return output_args_;
}

inline const SmallVector<Operation::StoreArg>& Operation::reduction_stores() const
{
  return reduction_args_;
}

}  // namespace legate::detail
