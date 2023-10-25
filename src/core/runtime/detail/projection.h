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

#include "core/utilities/tuple.h"
#include "core/utilities/typedefs.h"

#include <functional>
#include <iosfwd>

namespace legate::proj {

class SymbolicExpr {
 public:
  SymbolicExpr() = default;

  SymbolicExpr(int32_t dim, int32_t weight, int32_t offset = 0);

  explicit SymbolicExpr(int32_t dim);

  [[nodiscard]] int32_t dim() const;
  [[nodiscard]] int32_t weight() const;
  [[nodiscard]] int32_t offset() const;

  [[nodiscard]] bool is_identity(int32_t dim) const;

  bool operator==(const SymbolicExpr& other) const;
  bool operator<(const SymbolicExpr& other) const;

  SymbolicExpr operator*(int32_t other) const;
  SymbolicExpr operator+(int32_t other) const;

 private:
  int32_t dim_{-1};
  int32_t weight_{1};
  int32_t offset_{};
};

std::ostream& operator<<(std::ostream& out, const SymbolicExpr& expr);

using SymbolicPoint   = tuple<SymbolicExpr>;
using SymbolicFunctor = std::function<SymbolicPoint(const SymbolicPoint&)>;

struct RadixProjectionFunctor {
  RadixProjectionFunctor(int32_t radix, int32_t offset);

  [[nodiscard]] SymbolicPoint operator()(const SymbolicPoint& point) const;

 private:
  int32_t offset_{};
  int32_t radix_{};
};

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
