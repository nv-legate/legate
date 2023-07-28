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
#include <optional>

#include "legion.h"

#include "core/utilities/tuple.h"
#include "core/utilities/typedefs.h"

namespace legate::proj {

class SymbolicExpr {
 public:
  SymbolicExpr(int32_t dim = -1, int32_t weight = 1, int32_t offset = 0);

 public:
  int32_t dim() const { return dim_; }
  int32_t weight() const { return weight_; }
  int32_t offset() const { return offset_; }

 public:
  bool is_identity(int32_t dim) const;

 public:
  bool operator==(const SymbolicExpr& other) const;
  bool operator<(const SymbolicExpr& other) const;

 public:
  SymbolicExpr operator*(int32_t other) const;
  SymbolicExpr operator+(int32_t other) const;

 private:
  int32_t dim_{-1};
  int32_t weight_{1};
  int32_t offset_{0};
};

std::ostream& operator<<(std::ostream& out, const SymbolicExpr& expr);

using SymbolicPoint   = tuple<SymbolicExpr>;
using SymbolicFunctor = std::function<SymbolicPoint(const SymbolicPoint&)>;
;

struct RadixProjectionFunctor {
  RadixProjectionFunctor(int32_t radix, int32_t offset);

  SymbolicPoint operator()(const SymbolicPoint& exprs) const;

 private:
  int32_t offset_, radix_;
};

SymbolicPoint create_symbolic_point(int32_t ndim);

bool is_identity(int32_t ndim, const SymbolicPoint& point);

}  // namespace legate::proj

namespace legate::detail {

class Library;

// Interface for Legate projection functors
class LegateProjectionFunctor : public Legion::ProjectionFunctor {
 public:
  LegateProjectionFunctor(Legion::Runtime* runtime);

 public:
  using Legion::ProjectionFunctor::project;
  virtual Legion::LogicalRegion project(Legion::LogicalPartition upper_bound,
                                        const DomainPoint& point,
                                        const Domain& launch_domain);

 public:
  // legate projection functors are almost always functional and don't traverse the region tree
  virtual bool is_functional(void) const { return true; }
  virtual bool is_exclusive(void) const { return true; }
  virtual unsigned get_depth(void) const { return 0; }

 public:
  virtual DomainPoint project_point(const DomainPoint& point,
                                    const Domain& launch_domain) const = 0;
};

void register_legate_core_projection_functors(Legion::Runtime* runtime,
                                              const detail::Library* core_library);

LegateProjectionFunctor* find_legate_projection_functor(Legion::ProjectionID proj_id,
                                                        bool allow_missing = false);

}  // namespace legate::detail
