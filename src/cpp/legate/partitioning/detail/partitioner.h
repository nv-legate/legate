/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/detail/strategy.h>
#include <legate/utilities/span.h>

#include <vector>

namespace legate::detail {

class Operation;
class TransformStack;

class Partitioner {
 public:
  [[nodiscard]] static Strategy partition_stores(Operation* op);

  [[nodiscard]] static Legion::ProjectionID infer_store_projection(
    const Domain& launch_domain,
    Span<const std::uint64_t> color_shape,
    const InternalSharedPtr<TransformStack>& transform);
};

}  // namespace legate::detail
