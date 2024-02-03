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
#include "core/utilities/tuple.h"

#include <functional>
#include <iosfwd>

namespace legate::proj {

using SymbolicExpr  = legate::SymbolicExpr;
using SymbolicPoint = legate::SymbolicPoint;

[[nodiscard]] SymbolicPoint create_symbolic_point(uint32_t ndim);

[[nodiscard]] bool is_identity(uint32_t ndim, const SymbolicPoint& point);

}  // namespace legate::proj

namespace legate::detail {

class Library;

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

void register_affine_projection_functor(uint32_t src_ndim,
                                        const proj::SymbolicPoint& point,
                                        Legion::ProjectionID proj_id);

void register_delinearizing_projection_functor(const tuple<uint64_t>& color_shape,
                                               Legion::ProjectionID proj_id);

void register_compound_projection_functor(const tuple<uint64_t>& color_shape,
                                          const proj::SymbolicPoint& point,
                                          Legion::ProjectionID proj_id);

[[nodiscard]] ProjectionFunction* find_projection_function(Legion::ProjectionID proj_id) noexcept;

}  // namespace legate::detail
