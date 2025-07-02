/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/detail/strategy.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/span.h>

#include <vector>

namespace legate::detail {

class ConstraintSolver;
class Variable;

class Partitioner {
 public:
  explicit Partitioner(Span<const InternalSharedPtr<Operation>> operations);

  [[nodiscard]] Strategy partition_stores();

 private:
  // Populates solutions for unbound stores in the `strategy` and returns remaining partition
  // symbols
  [[nodiscard]] static std::vector<const Variable*> handle_unbound_stores_(
    Span<const Variable* const> partition_symbols,
    const ConstraintSolver& solver,
    Strategy* strategy);

  Span<const InternalSharedPtr<Operation>> operations_{};
};

}  // namespace legate::detail

#include <legate/partitioning/detail/partitioner.inl>
