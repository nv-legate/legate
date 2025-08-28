/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/projection.h>

#include <legate/runtime/detail/library.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/dispatch.h>
#include <legate/utilities/typedefs.h>

#include <fmt/format.h>

#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace legate::proj {

SymbolicPoint create_symbolic_point(std::uint32_t ndim)
{
  std::vector<SymbolicExpr> exprs;

  exprs.reserve(ndim);
  for (std::uint32_t dim = 0; dim < ndim; ++dim) {
    exprs.emplace_back(dim);
  }
  return SymbolicPoint{std::move(exprs)};
}

bool is_identity(std::uint32_t src_ndim, const SymbolicPoint& point)
{
  auto ndim = static_cast<std::uint32_t>(point.size());
  if (src_ndim != ndim) {
    return false;
  }
  for (std::uint32_t dim = 0; dim < ndim; ++dim) {
    if (!point[dim].is_identity(dim)) {
      return false;
    }
  }
  return true;
}

}  // namespace legate::proj

namespace legate::detail {

// ==========================================================================================

template <std::int32_t SRC_NDIM, std::int32_t TGT_NDIM>
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

template <std::int32_t SRC_NDIM, std::int32_t TGT_NDIM>
AffineProjection<SRC_NDIM, TGT_NDIM>::AffineProjection(const proj::SymbolicPoint& point)
  : transform_{create_transform(point)}
{
  for (std::int32_t dim = 0; dim < TGT_NDIM; ++dim) {
    offsets_[dim] = point[dim].offset();
  }
}

template <std::int32_t SRC_NDIM, std::int32_t TGT_NDIM>
DomainPoint AffineProjection<SRC_NDIM, TGT_NDIM>::project_point(const DomainPoint& point) const
{
  return DomainPoint{transform_ * Point<SRC_NDIM>{point} + offsets_};
}

template <std::int32_t SRC_NDIM, std::int32_t TGT_NDIM>
/*static*/ Legion::Transform<TGT_NDIM, SRC_NDIM>
AffineProjection<SRC_NDIM, TGT_NDIM>::create_transform(const proj::SymbolicPoint& point)
{
  Legion::Transform<TGT_NDIM, SRC_NDIM> transform;

  for (std::int32_t tgt_dim = 0; tgt_dim < TGT_NDIM; ++tgt_dim) {
    for (std::int32_t src_dim = 0; src_dim < SRC_NDIM; ++src_dim) {
      transform[tgt_dim][src_dim] = 0;
    }
  }

  for (std::int32_t tgt_dim = 0; tgt_dim < TGT_NDIM; ++tgt_dim) {
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
  explicit DelinearizingProjection(const SmallVector<std::uint64_t, LEGATE_MAX_DIM>& color_shape);

  [[nodiscard]] DomainPoint project_point(const DomainPoint& point) const override;

 private:
  std::vector<std::int64_t> strides_{};
};

DelinearizingProjection::DelinearizingProjection(
  const SmallVector<std::uint64_t, LEGATE_MAX_DIM>& color_shape)
  : strides_(color_shape.size(), 1)
{
  for (std::uint32_t dim = color_shape.size() - 1; dim > 0; --dim) {
    strides_[dim - 1] = strides_[dim] * static_cast<std::int64_t>(color_shape[dim]);
  }
}

DomainPoint DelinearizingProjection::project_point(const DomainPoint& point) const
{
  LEGATE_ASSERT(point.dim == 1);

  DomainPoint result;
  std::int64_t value = point[0];

  result.dim = static_cast<std::int32_t>(strides_.size());
  for (std::int32_t dim = 0; dim < result.dim; ++dim) {
    result[dim] = value / strides_[dim];
    value       = value % strides_[dim];
  }

  return result;
}

// ==========================================================================================

template <std::int32_t SRC_NDIM, std::int32_t TGT_NDIM>
class CompoundProjection final : public ProjectionFunction {
 public:
  CompoundProjection(const SmallVector<std::uint64_t, LEGATE_MAX_DIM>& color_shape,
                     const proj::SymbolicPoint& point);

  [[nodiscard]] DomainPoint project_point(const DomainPoint& point) const override;

 private:
  DelinearizingProjection delinearizing_projection_;
  AffineProjection<SRC_NDIM, TGT_NDIM> affine_projection_;
};

template <std::int32_t SRC_NDIM, std::int32_t TGT_NDIM>
CompoundProjection<SRC_NDIM, TGT_NDIM>::CompoundProjection(
  const SmallVector<std::uint64_t, LEGATE_MAX_DIM>& color_shape, const proj::SymbolicPoint& point)
  : delinearizing_projection_{color_shape}, affine_projection_{point}
{
}

template <std::int32_t SRC_NDIM, std::int32_t TGT_NDIM>
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

class RegisterAffineFunctorFn {
 public:
  template <std::int32_t SRC_NDIM, std::int32_t TGT_NDIM>
  void operator()(const proj::SymbolicPoint& point, Legion::ProjectionID proj_id)
  {
    register_legion_functor(proj_id, std::make_unique<AffineProjection<SRC_NDIM, TGT_NDIM>>(point));
  }
};

class RegisterCompoundFunctorFn {
 public:
  template <std::int32_t SRC_NDIM, std::int32_t TGT_NDIM>
  void operator()(const SmallVector<std::uint64_t, LEGATE_MAX_DIM>& color_shape,
                  const proj::SymbolicPoint& point,
                  Legion::ProjectionID proj_id)
  {
    register_legion_functor(
      proj_id, std::make_unique<CompoundProjection<SRC_NDIM, TGT_NDIM>>(color_shape, point));
  }
};

}  // namespace

ProjectionFunction* find_projection_function(Legion::ProjectionID proj_id)
{
  if (0 == proj_id) {
    return identity_projection();
  }

  const std::lock_guard<std::mutex> lock{functor_table_lock};
  auto finder = functor_table.find(proj_id);

  if (finder == functor_table.end()) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Failed to find projection functor of id {}", proj_id)};
  }

  return finder->second.get();
}

void register_affine_projection_functor(std::uint32_t src_ndim,
                                        const proj::SymbolicPoint& point,
                                        Legion::ProjectionID proj_id)
{
  legate::double_dispatch(static_cast<std::int32_t>(src_ndim),
                          static_cast<std::int32_t>(point.size()),
                          RegisterAffineFunctorFn{},
                          point,
                          proj_id);
}

void register_delinearizing_projection_functor(
  const SmallVector<std::uint64_t, LEGATE_MAX_DIM>& color_shape, Legion::ProjectionID proj_id)
{
  register_legion_functor(proj_id, std::make_unique<DelinearizingProjection>(color_shape));
}

void register_compound_projection_functor(
  const SmallVector<std::uint64_t, LEGATE_MAX_DIM>& color_shape,
  const proj::SymbolicPoint& point,
  Legion::ProjectionID proj_id)
{
  legate::double_dispatch(static_cast<std::int32_t>(color_shape.size()),
                          static_cast<std::int32_t>(point.size()),
                          RegisterCompoundFunctorFn{},
                          color_shape,
                          point,
                          proj_id);
}

class LinearizingPointTransformFunctor final : public Legion::PointTransformFunctor {
 public:
  // This is actually an invertible functor, but we will not use this for inversion
  [[nodiscard]] bool is_invertible() const override { return false; }

  DomainPoint transform_point(const DomainPoint& point,
                              const Domain& domain,
                              const Domain& range) override
  {
    LEGATE_CHECK(range.dim == 1);
    DomainPoint result;
    result.dim = 1;

    const std::int32_t ndim = domain.dim;
    std::int64_t idx        = point[0];
    for (std::int32_t dim = 1; dim < ndim; ++dim) {
      const std::int64_t extent = domain.rect_data[dim + ndim] - domain.rect_data[dim] + 1;
      idx                       = idx * extent + point[dim];
    }
    result[0] = idx;
    return result;
  }
};

}  // namespace legate::detail
