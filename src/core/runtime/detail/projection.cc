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

SymbolicPoint create_symbolic_point(int32_t ndim)
{
  std::vector<SymbolicExpr> exprs;

  exprs.reserve(ndim);
  for (int32_t dim = 0; dim < ndim; ++dim) {
    exprs.emplace_back(dim);
  }
  return SymbolicPoint{std::move(exprs)};
}

bool is_identity(int32_t src_ndim, const SymbolicPoint& point)
{
  auto ndim = static_cast<int32_t>(point.size());
  if (src_ndim != ndim) {
    return false;
  }
  for (int32_t dim = 0; dim < ndim; ++dim) {
    if (!point[dim].is_identity(dim)) {
      return false;
    }
  }
  return true;
}

}  // namespace legate::proj

namespace legate::detail {

// This special functor overrides the default projection implementation because it needs
// to know the the target color space for delinearization. Also note that this functor's
// project_point passes through input points, as we already know they are always 1D points
// and the output will be linearized back to integers.
class DelinearizationFunctor final : public LegateProjectionFunctor {
 public:
  using LegateProjectionFunctor::LegateProjectionFunctor;

  [[nodiscard]] Legion::LogicalRegion project(Legion::LogicalPartition upper_bound,
                                              const DomainPoint& point,
                                              const Domain& launch_domain) override;

  [[nodiscard]] DomainPoint project_point(const DomainPoint& point,
                                          const Domain& launch_domain) const override;
};

template <int32_t SRC_DIM, int32_t TGT_DIM>
class AffineFunctor : public LegateProjectionFunctor {
 public:
  AffineFunctor(Legion::Runtime* lg_runtime, const proj::SymbolicPoint& point);

  [[nodiscard]] DomainPoint project_point(const DomainPoint& point,
                                          const Domain& launch_domain) const override;

  [[nodiscard]] static Legion::Transform<TGT_DIM, SRC_DIM> create_transform(
    const proj::SymbolicPoint& point);

 private:
  Legion::Transform<TGT_DIM, SRC_DIM> transform_{};
  Point<TGT_DIM> offsets_{};
};

Legion::LogicalRegion LegateProjectionFunctor::project(Legion::LogicalPartition upper_bound,
                                                       const DomainPoint& point,
                                                       const Domain& launch_domain)
{
  const auto dp = project_point(point, launch_domain);
  if (runtime->has_logical_subregion_by_color(upper_bound, dp)) {
    return runtime->get_logical_subregion_by_color(upper_bound, dp);
  }
  return Legion::LogicalRegion::NO_REGION;
}

Legion::LogicalRegion DelinearizationFunctor::project(Legion::LogicalPartition upper_bound,
                                                      const DomainPoint& point,
                                                      const Domain& /*launch_domain*/)
{
  const auto color_space =
    runtime->get_index_partition_color_space(upper_bound.get_index_partition());

  assert(color_space.dense());
  assert(point.dim == 1);

  std::vector<int64_t> strides(color_space.dim, 1);
  for (int32_t dim = color_space.dim - 1; dim > 0; --dim) {
    auto extent = color_space.rect_data[dim + color_space.dim] - color_space.rect_data[dim] + 1;
    strides[dim - 1] = strides[dim] * extent;
  }

  DomainPoint delinearized;
  int64_t value = point[0];

  delinearized.dim = color_space.dim;
  for (int32_t dim = 0; dim < color_space.dim; ++dim) {
    delinearized[dim] = value / strides[dim];
    value             = value % strides[dim];
  }

  if (runtime->has_logical_subregion_by_color(upper_bound, delinearized)) {
    return runtime->get_logical_subregion_by_color(upper_bound, delinearized);
  }
  return Legion::LogicalRegion::NO_REGION;
}

DomainPoint DelinearizationFunctor::project_point(const DomainPoint& point,
                                                  const Domain& /*launch_domain*/) const
{
  return point;
}

template <int32_t SRC_NDIM, int32_t TGT_NDIM>
AffineFunctor<SRC_NDIM, TGT_NDIM>::AffineFunctor(Legion::Runtime* lg_runtime,
                                                 const proj::SymbolicPoint& point)
  : LegateProjectionFunctor{lg_runtime}, transform_{create_transform(point)}

{
  for (int32_t dim = 0; dim < TGT_NDIM; ++dim) {
    offsets_[dim] = point[dim].offset();
  }
}

template <int32_t SRC_NDIM, int32_t TGT_NDIM>
DomainPoint AffineFunctor<SRC_NDIM, TGT_NDIM>::project_point(const DomainPoint& point,
                                                             const Domain& /*launch_domain*/) const
{
  return DomainPoint{transform_ * Point<SRC_NDIM>{point} + offsets_};
}

template <int32_t SRC_NDIM, int32_t TGT_NDIM>
/*static*/ Legion::Transform<TGT_NDIM, SRC_NDIM>
AffineFunctor<SRC_NDIM, TGT_NDIM>::create_transform(const proj::SymbolicPoint& point)
{
  Legion::Transform<TGT_NDIM, SRC_NDIM> transform;

  for (int32_t tgt_dim = 0; tgt_dim < TGT_NDIM; ++tgt_dim) {
    for (int32_t src_dim = 0; src_dim < SRC_NDIM; ++src_dim) {
      transform[tgt_dim][src_dim] = 0;
    }
  }

  for (int32_t tgt_dim = 0; tgt_dim < TGT_NDIM; ++tgt_dim) {
    const auto& expr = point[tgt_dim];
    if (expr.dim() != -1) {
      transform[tgt_dim][expr.dim()] = expr.weight();
    }
  }

  return transform;
}

struct IdentityFunctor final : public LegateProjectionFunctor {
  using LegateProjectionFunctor::LegateProjectionFunctor;

  [[nodiscard]] DomainPoint project_point(const DomainPoint& point, const Domain&) const override
  {
    return point;
  }
};

namespace {

LegateProjectionFunctor* identity_functor{};
std::unordered_map<Legion::ProjectionID, LegateProjectionFunctor*> functor_table{};
std::mutex functor_table_lock{};

}  // namespace

struct register_affine_functor_fn {
  template <int32_t SRC_NDIM, int32_t TGT_NDIM>
  static void spec_to_string(std::stringstream& ss, const proj::SymbolicPoint& point)
  {
    ss << "\\(";
    for (int32_t idx = 0; idx < SRC_NDIM; ++idx) {
      if (idx != 0) {
        ss << ",";
      }
      ss << "x" << idx;
    }
    ss << ")->(";
    for (int32_t idx = 0; idx < TGT_NDIM; ++idx) {
      auto& expr = point[idx];

      if (idx != 0) {
        ss << ",";
      }
      if (expr.dim() != -1) {
        if (expr.weight() != 0) {
          if (expr.weight() != 1) {
            ss << expr.weight() << "*";
          }
          ss << "x" << expr.dim();
        }
      }
      if (expr.offset() != 0) {
        if (expr.offset() > 0) {
          ss << "+" << expr.offset();
        } else {
          ss << "-" << -expr.offset();
        }
      } else if (expr.weight() == 0) {
        ss << "0";
      }
    }
    ss << ")";
  }

  template <int32_t SRC_NDIM, int32_t TGT_NDIM>
  void operator()(const proj::SymbolicPoint& point, Legion::ProjectionID proj_id)
  {
    auto runtime = Legion::Runtime::get_runtime();
    auto functor = new AffineFunctor<SRC_NDIM, TGT_NDIM>{runtime, point};

    if (LegateDefined(LEGATE_USE_DEBUG)) {
      std::stringstream ss;

      ss << "Register projection functor: functor: " << functor << ", id: " << proj_id << ", ";
      spec_to_string<SRC_NDIM, TGT_NDIM>(ss, point);
      log_legate().debug() << std::move(ss).str();
    } else {
      log_legate().debug(
        "Register projection functor: functor: %p, id: %d", static_cast<void*>(functor), proj_id);
    }
    runtime->register_projection_functor(proj_id, functor, true /*silence warnings*/);

    const std::lock_guard<std::mutex> lock{functor_table_lock};

    functor_table[proj_id] = functor;
  }
};

void register_legate_core_projection_functors(const detail::Library* core_library)
{
  const auto runtime = Legion::Runtime::get_runtime();
  auto proj_id       = core_library->get_projection_id(LEGATE_CORE_DELINEARIZE_PROJ_ID);
  auto functor       = new DelinearizationFunctor{runtime};

  log_legate().debug(
    "Register delinearizing functor: functor: %p, id: %d", static_cast<void*>(functor), proj_id);
  runtime->register_projection_functor(proj_id, functor, true /*silence warnings*/);
  {
    const std::lock_guard<std::mutex> lock{functor_table_lock};

    functor_table[proj_id] = functor;
  }
  identity_functor = new IdentityFunctor{runtime};
}

LegateProjectionFunctor* find_legate_projection_functor(Legion::ProjectionID proj_id,
                                                        bool allow_missing)
{
  if (0 == proj_id) {
    return identity_functor;
  }

  const std::lock_guard<std::mutex> lock{functor_table_lock};
  auto result = functor_table[proj_id];

  // If we're not OK with a missing projection functor, then throw an error.
  if (nullptr == result && !allow_missing) {
    log_legate().debug("Failed to find projection functor of id %d", proj_id);
    LEGATE_ABORT;
  }
  return result;
}

void register_affine_projection_functor(int32_t src_ndim,
                                        const proj::SymbolicPoint& point,
                                        legion_projection_id_t proj_id)
{
  legate::double_dispatch(
    src_ndim, static_cast<int32_t>(point.size()), register_affine_functor_fn{}, point, proj_id);
}

struct LinearizingPointTransformFunctor final : public Legion::PointTransformFunctor {
  // This is actually an invertible functor, but we will not use this for inversion
  [[nodiscard]] bool is_invertible() const override { return false; }

  DomainPoint transform_point(const DomainPoint& point,
                              const Domain& domain,
                              const Domain& range) override
  {
    assert(range.dim == 1);
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
