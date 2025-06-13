/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/projection.h>
#include <legate/utilities/tuple.h>

namespace legate::proj {

using legate::SymbolicExpr;
using legate::SymbolicPoint;

[[nodiscard]] SymbolicPoint create_symbolic_point(std::uint32_t ndim);

[[nodiscard]] bool is_identity(std::uint32_t ndim, const SymbolicPoint& point);

}  // namespace legate::proj

namespace legate::detail {

class ProjectionFunction {  // NOLINT(bugprone-forward-declaration-namespace)
 public:
  [[nodiscard]] virtual DomainPoint project_point(const DomainPoint& point) const = 0;

  virtual ~ProjectionFunction()                            = default;
  ProjectionFunction()                                     = default;
  ProjectionFunction(const ProjectionFunction&)            = default;
  ProjectionFunction(ProjectionFunction&&)                 = default;
  ProjectionFunction& operator=(const ProjectionFunction&) = default;
  ProjectionFunction& operator=(ProjectionFunction&&)      = default;
};

void register_affine_projection_functor(std::uint32_t src_ndim,
                                        const proj::SymbolicPoint& point,
                                        Legion::ProjectionID proj_id);

void register_delinearizing_projection_functor(const tuple<std::uint64_t>& color_shape,
                                               Legion::ProjectionID proj_id);

void register_compound_projection_functor(const tuple<std::uint64_t>& color_shape,
                                          const proj::SymbolicPoint& point,
                                          Legion::ProjectionID proj_id);

[[nodiscard]] ProjectionFunction* find_projection_function(Legion::ProjectionID proj_id);

}  // namespace legate::detail
