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

#include "core/runtime/detail/projection.h"

#include "core/runtime/detail/library.h"
#include "core/utilities/dispatch.h"
#include "core/utilities/typedefs.h"

#include <iostream>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace legate::proj {

SymbolicPoint create_symbolic_point(uint32_t ndim)
{
  std::vector<SymbolicExpr> exprs;

  exprs.reserve(ndim);
  for (uint32_t dim = 0; dim < ndim; ++dim) {
    exprs.emplace_back(dim);
  }
  return SymbolicPoint{std::move(exprs)};
}

bool is_identity(uint32_t src_ndim, const SymbolicPoint& point)
{
  auto ndim = static_cast<uint32_t>(point.size());
  if (src_ndim != ndim) {
    return false;
  }
  for (uint32_t dim = 0; dim < ndim; ++dim) {
    if (!point[dim].is_identity(dim)) {
      return false;
    }
  }
  return true;
}

}  // namespace legate::proj

namespace legate::detail {

// ==========================================================================================

template <int32_t SRC_NDIM, int32_t TGT_NDIM>
class AffineProjection final : public ProjectionFunction {
 public:
  explicit AffineProjection(const proj::SymbolicPoint& point);

  [[nodiscard]] DomainPoint project_point(const DomainPoint& point) const override;

  [[nodiscard]] static Legion::Transform<TGT_NDIM, SRC_NDIM> create_transform(
    const proj::SymbolicPoint& point);

 private:
  Legion::Transform<TGT_NDIM, SRC_NDIM> transform_{};
  Point<TGT_NDIM> offsets_{};
};

template <int32_t SRC_NDIM, int32_t TGT_NDIM>
AffineProjection<SRC_NDIM, TGT_NDIM>::AffineProjection(const proj::SymbolicPoint& point)
  : transform_{create_transform(point)}
{
  for (int32_t dim = 0; dim < TGT_NDIM; ++dim) {
    offsets_[dim] = point[dim].offset();
  }
}

template <int32_t SRC_NDIM, int32_t TGT_NDIM>
DomainPoint AffineProjection<SRC_NDIM, TGT_NDIM>::project_point(const DomainPoint& point) const
{
  return DomainPoint{transform_ * Point<SRC_NDIM>{point} + offsets_};
}

template <int32_t SRC_NDIM, int32_t TGT_NDIM>
/*static*/ Legion::Transform<TGT_NDIM, SRC_NDIM>
AffineProjection<SRC_NDIM, TGT_NDIM>::create_transform(const proj::SymbolicPoint& point)
{
  Legion::Transform<TGT_NDIM, SRC_NDIM> transform;

  for (int32_t tgt_dim = 0; tgt_dim < TGT_NDIM; ++tgt_dim) {
    for (int32_t src_dim = 0; src_dim < SRC_NDIM; ++src_dim) {
      transform[tgt_dim][src_dim] = 0;
    }
  }

  for (int32_t tgt_dim = 0; tgt_dim < TGT_NDIM; ++tgt_dim) {
    const auto& expr = point[tgt_dim];
    if (!expr.is_constant()) {
      transform[tgt_dim][expr.dim()] = expr.weight();
    }
  }

  return transform;
}

// ==========================================================================================

class DelinearizingProjection final : public ProjectionFunction {
 public:
  explicit DelinearizingProjection(const tuple<uint64_t>& color_shape);

  [[nodiscard]] DomainPoint project_point(const DomainPoint& point) const override;

 private:
  std::vector<int64_t> strides{};
};

DelinearizingProjection::DelinearizingProjection(const tuple<uint64_t>& color_shape)
  : strides(color_shape.size(), 1)
{
  for (uint32_t dim = color_shape.size() - 1; dim > 0; --dim) {
    strides[dim - 1] = strides[dim] * static_cast<int64_t>(color_shape[dim]);
  }
}

DomainPoint DelinearizingProjection::project_point(const DomainPoint& point) const
{
  LegateAssert(point.dim == 1);

  DomainPoint result;
  int64_t value = point[0];

  result.dim = static_cast<int32_t>(strides.size());
  for (int32_t dim = 0; dim < result.dim; ++dim) {
    result[dim] = value / strides[dim];
    value       = value % strides[dim];
  }

  return result;
}

// ==========================================================================================

template <int32_t SRC_NDIM, int32_t TGT_NDIM>
class CompoundProjection final : public ProjectionFunction {
 public:
  CompoundProjection(const tuple<uint64_t>& color_shape, const proj::SymbolicPoint& point);

  [[nodiscard]] DomainPoint project_point(const DomainPoint& point) const override;

 private:
  DelinearizingProjection delinearizing_projection_;
  AffineProjection<SRC_NDIM, TGT_NDIM> affine_projection_;
};

template <int32_t SRC_NDIM, int32_t TGT_NDIM>
CompoundProjection<SRC_NDIM, TGT_NDIM>::CompoundProjection(const tuple<uint64_t>& color_shape,
                                                           const proj::SymbolicPoint& point)
  : delinearizing_projection_{color_shape}, affine_projection_{point}
{
}

template <int32_t SRC_NDIM, int32_t TGT_NDIM>
DomainPoint CompoundProjection<SRC_NDIM, TGT_NDIM>::project_point(const DomainPoint& point) const
{
  return affine_projection_.project_point(delinearizing_projection_.project_point(point));
}

// ==========================================================================================

class IdentityProjection final : public ProjectionFunction {
 public:
  [[nodiscard]] DomainPoint project_point(const DomainPoint& point) const override { return point; }
};

// ==========================================================================================

class LegateProjectionFunctor : public Legion::ProjectionFunctor {
 public:
  LegateProjectionFunctor(Legion::Runtime* legion_runtime, ProjectionFunction* functor);

  using Legion::ProjectionFunctor::project;
  [[nodiscard]] Legion::LogicalRegion project(Legion::LogicalPartition upper_bound,
                                              const DomainPoint& point,
                                              const Domain& launch_domain) override;

  // legate projection functors are almost always functional and don't traverse the region tree
  [[nodiscard]] bool is_functional() const override { return true; }
  [[nodiscard]] bool is_exclusive() const override { return true; }
  [[nodiscard]] unsigned get_depth() const override { return 0; }

 private:
  ProjectionFunction* functor_{};
};

LegateProjectionFunctor::LegateProjectionFunctor(Legion::Runtime* legion_runtime,
                                                 ProjectionFunction* functor)
  : ProjectionFunctor{legion_runtime}, functor_{functor}
{
}

Legion::LogicalRegion LegateProjectionFunctor::project(Legion::LogicalPartition upper_bound,
                                                       const DomainPoint& point,
                                                       const Domain& /*launch_domain*/)
{
  const auto dp = functor_->project_point(point);
  if (runtime->has_logical_subregion_by_color(upper_bound, dp)) {
    return runtime->get_logical_subregion_by_color(upper_bound, dp);
  }
  return Legion::LogicalRegion::NO_REGION;
}

// ==========================================================================================

namespace {

ProjectionFunction* identity_projection()
{
  static auto identity_projection = std::make_unique<IdentityProjection>();
  return identity_projection.get();
}

std::unordered_map<Legion::ProjectionID, std::unique_ptr<ProjectionFunction>> functor_table{};
std::mutex functor_table_lock{};

void register_legion_functor(Legion::ProjectionID proj_id,
                             std::unique_ptr<ProjectionFunction> legate_functor)
{
  auto runtime = Legion::Runtime::get_runtime();
  runtime->register_projection_functor(
    proj_id, new LegateProjectionFunctor{runtime, legate_functor.get()}, true /*silence warnings*/);

  const std::lock_guard<std::mutex> lock{functor_table_lock};

  functor_table.try_emplace(proj_id, std::move(legate_functor));
}

struct register_affine_functor_fn {
  template <int32_t SRC_NDIM, int32_t TGT_NDIM>
  void operator()(const proj::SymbolicPoint& point, Legion::ProjectionID proj_id)
  {
    register_legion_functor(proj_id, std::make_unique<AffineProjection<SRC_NDIM, TGT_NDIM>>(point));
  }
};

struct register_compound_functor_fn {
  template <int32_t SRC_NDIM, int32_t TGT_NDIM>
  void operator()(const tuple<uint64_t>& color_shape,
                  const proj::SymbolicPoint& point,
                  Legion::ProjectionID proj_id)
  {
    register_legion_functor(
      proj_id, std::make_unique<CompoundProjection<SRC_NDIM, TGT_NDIM>>(color_shape, point));
  }
};

}  // namespace

ProjectionFunction* find_projection_function(Legion::ProjectionID proj_id) noexcept
{
  if (0 == proj_id) {
    return identity_projection();
  }

  const std::lock_guard<std::mutex> lock{functor_table_lock};
  auto finder = functor_table.find(proj_id);

  if (finder == functor_table.end()) {
    LEGATE_ABORT("Failed to find projection functor of id " << proj_id);
  }

  return finder->second.get();
}

void register_affine_projection_functor(uint32_t src_ndim,
                                        const proj::SymbolicPoint& point,
                                        Legion::ProjectionID proj_id)
{
  legate::double_dispatch(static_cast<int32_t>(src_ndim),
                          static_cast<int32_t>(point.size()),
                          register_affine_functor_fn{},
                          point,
                          proj_id);
}

void register_delinearizing_projection_functor(const tuple<uint64_t>& color_shape,
                                               Legion::ProjectionID proj_id)
{
  register_legion_functor(proj_id, std::make_unique<DelinearizingProjection>(color_shape));
}

void register_compound_projection_functor(const tuple<uint64_t>& color_shape,
                                          const proj::SymbolicPoint& point,
                                          Legion::ProjectionID proj_id)
{
  legate::double_dispatch(static_cast<int32_t>(color_shape.size()),
                          static_cast<int32_t>(point.size()),
                          register_compound_functor_fn{},
                          color_shape,
                          point,
                          proj_id);
}

struct LinearizingPointTransformFunctor final : public Legion::PointTransformFunctor {
  // This is actually an invertible functor, but we will not use this for inversion
  [[nodiscard]] bool is_invertible() const override { return false; }

  DomainPoint transform_point(const DomainPoint& point,
                              const Domain& domain,
                              const Domain& range) override
  {
    LegateCheck(range.dim == 1);
    DomainPoint result;
    result.dim = 1;

    const int32_t ndim = domain.dim;
    int64_t idx        = point[0];
    for (int32_t dim = 1; dim < ndim; ++dim) {
      const int64_t extent = domain.rect_data[dim + ndim] - domain.rect_data[dim] + 1;
      idx                  = idx * extent + point[dim];
    }
    result[0] = idx;
    return result;
  }
};

}  // namespace legate::detail
