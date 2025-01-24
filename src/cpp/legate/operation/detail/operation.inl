/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <legate/operation/detail/operation.h>

namespace legate::detail {

inline bool Operation::StoreArg::needs_flush() const { return store->needs_flush(); }

// ==========================================================================================

inline void Operation::validate() {}

inline void Operation::add_to_solver(ConstraintSolver& /*solver*/) {}

inline void Operation::launch() { LEGATE_ABORT("This method should have been overriden"); }

inline void Operation::launch(Strategy* /*strategy*/)
{
  LEGATE_ABORT("This method should have been overriden");
}

inline bool Operation::supports_replicated_write() const { return false; }

inline std::int32_t Operation::priority() const { return priority_; }

inline const mapping::detail::Machine& Operation::machine() const { return machine_; }

inline ZStringView Operation::provenance() const { return provenance_; }

}  // namespace legate::detail
