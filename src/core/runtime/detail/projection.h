/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include "core/operation/projection.h"

#include <functional>
#include <iosfwd>

namespace legate::proj {

using SymbolicExpr  = legate::SymbolicExpr;
using SymbolicPoint = legate::SymbolicPoint;

[[nodiscard]] SymbolicPoint create_symbolic_point(int32_t ndim);

[[nodiscard]] bool is_identity(int32_t ndim, const SymbolicPoint& point);

}  // namespace legate::proj

namespace legate::detail {

class Library;

// Interface for Legate projection functors
class LegateProjectionFunctor : public Legion::ProjectionFunctor {
 public:
  explicit LegateProjectionFunctor(Legion::Runtime* runtime);

  using Legion::ProjectionFunctor::project;
  [[nodiscard]] Legion::LogicalRegion project(Legion::LogicalPartition upper_bound,
                                              const DomainPoint& point,
                                              const Domain& launch_domain) override;

  // legate projection functors are almost always functional and don't traverse the region tree
  [[nodiscard]] bool is_functional() const override;
  [[nodiscard]] bool is_exclusive() const override;
  [[nodiscard]] unsigned get_depth() const override;

  [[nodiscard]] virtual DomainPoint project_point(const DomainPoint& point,
                                                  const Domain& launch_domain) const = 0;
};

void register_legate_core_projection_functors(Legion::Runtime* runtime,
                                              const detail::Library* core_library);

[[nodiscard]] LegateProjectionFunctor* find_legate_projection_functor(Legion::ProjectionID proj_id,
                                                                      bool allow_missing = false);

}  // namespace legate::detail

#include "core/runtime/detail/projection.inl"
