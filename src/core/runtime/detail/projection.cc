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
  AffineFunctor(Legion::Runtime* lg_runtime,
                int32_t* dims,
                int32_t* weights,
                const int32_t* offsets);

  [[nodiscard]] DomainPoint project_point(const DomainPoint& point,
                                          const Domain& launch_domain) const override;

  [[nodiscard]] static Legion::Transform<TGT_DIM, SRC_DIM> create_transform(const int32_t* dims,
                                                                            const int32_t* weights);

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

template <int32_t SRC_DIM, int32_t TGT_DIM>
AffineFunctor<SRC_DIM, TGT_DIM>::AffineFunctor(Legion::Runtime* lg_runtime,
                                               int32_t* dims,
                                               int32_t* weights,
                                               const int32_t* offsets)
  : LegateProjectionFunctor{lg_runtime}, transform_{create_transform(dims, weights)}

{
  for (int32_t dim = 0; dim < TGT_DIM; ++dim) {
    offsets_[dim] = offsets[dim];
  }
}

template <int32_t SRC_DIM, int32_t TGT_DIM>
DomainPoint AffineFunctor<SRC_DIM, TGT_DIM>::project_point(const DomainPoint& point,
                                                           const Domain& /*launch_domain*/) const
{
  return DomainPoint{transform_ * Point<SRC_DIM>{point} + offsets_};
}

template <int32_t SRC_DIM, int32_t TGT_DIM>
/*static*/ Legion::Transform<TGT_DIM, SRC_DIM> AffineFunctor<SRC_DIM, TGT_DIM>::create_transform(
  const int32_t* dims, const int32_t* weights)
{
  Legion::Transform<TGT_DIM, SRC_DIM> transform;

  for (int32_t tgt_dim = 0; tgt_dim < TGT_DIM; ++tgt_dim) {
    for (int32_t src_dim = 0; src_dim < SRC_DIM; ++src_dim) {
      transform[tgt_dim][src_dim] = 0;
    }
  }

  for (int32_t tgt_dim = 0; tgt_dim < TGT_DIM; ++tgt_dim) {
    const int32_t src_dim = dims[tgt_dim];
    if (src_dim != -1) {
      transform[tgt_dim][src_dim] = weights[tgt_dim];
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

struct create_affine_functor_fn {
  static void spec_to_string(std::stringstream& ss,
                             int32_t src_ndim,
                             int32_t tgt_ndim,
                             const int32_t* dims,
                             const int32_t* weights,
                             const int32_t* offsets)
  {
    ss << "\\(";
    for (int32_t idx = 0; idx < src_ndim; ++idx) {
      if (idx != 0) {
        ss << ",";
      }
      ss << "x" << idx;
    }
    ss << ")->(";
    for (int32_t idx = 0; idx < tgt_ndim; ++idx) {
      auto dim    = dims[idx];
      auto weight = weights[idx];
      auto offset = offsets[idx];

      if (idx != 0) {
        ss << ",";
      }
      if (dim != -1) {
        if (weight != 0) {
          assert(dim != -1);
          if (weight != 1) {
            ss << weight << "*";
          }
          ss << "x" << dim;
        }
      }
      if (offset != 0) {
        if (offset > 0) {
          ss << "+" << offset;
        } else {
          ss << "-" << -offset;
        }
      } else if (weight == 0) {
        ss << "0";
      }
    }
    ss << ")";
  }

  template <int32_t SRC_DIM, int32_t TGT_DIM>
  void operator()(Legion::Runtime* runtime,
                  int32_t* dims,
                  int32_t* weights,
                  int32_t* offsets,
                  Legion::ProjectionID proj_id)
  {
    auto functor = new AffineFunctor<SRC_DIM, TGT_DIM>{runtime, dims, weights, offsets};

    if (LegateDefined(LEGATE_USE_DEBUG)) {
      std::stringstream ss;

      ss << "Register projection functor: functor: " << functor << ", id: " << proj_id << ", ";
      spec_to_string(ss, SRC_DIM, TGT_DIM, dims, weights, offsets);
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

void register_legate_core_projection_functors(Legion::Runtime* runtime,
                                              const detail::Library* core_library)
{
  auto proj_id = core_library->get_projection_id(LEGATE_CORE_DELINEARIZE_PROJ_ID);
  auto functor = new DelinearizationFunctor{runtime};

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

extern "C" {

void legate_register_affine_projection_functor(int32_t src_ndim,
                                               int32_t tgt_ndim,
                                               int32_t* dims,
                                               int32_t* weights,
                                               int32_t* offsets,
                                               legion_projection_id_t proj_id)
{
  auto runtime = Legion::Runtime::get_runtime();
  legate::double_dispatch(src_ndim,
                          tgt_ndim,
                          legate::detail::create_affine_functor_fn{},
                          runtime,
                          dims,
                          weights,
                          offsets,
                          proj_id);
}

[[nodiscard]] void* legate_linearizing_point_transform_functor()
{
  try {
    static auto* functor = new legate::detail::LinearizingPointTransformFunctor{};

    return functor;
  } catch (...) {
    std::terminate();
  }
}
//
}
