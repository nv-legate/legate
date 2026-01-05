/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/proxy.h>
#include <legate/task/task.h>
#include <legate/task/task_config.h>
#include <legate/task/task_signature.h>
#include <legate/task/variant_options.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>

namespace legate::detail {

class Library;

class LEGATE_EXPORT FindBoundingBox : public LegateTask<FindBoundingBox> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp
    legate::TaskConfig{LocalTaskID{CoreTask::FIND_BOUNDING_BOX}}.with_signature(
      legate::TaskSignature{}.inputs(1).outputs(1).scalars(0).redops(0).constraints(
        {Span<const legate::ProxyConstraint>{}})  // some compilers complain with {{}}
    );
  static constexpr VariantOptions GPU_VARIANT_OPTIONS =
    VariantOptions{}.with_elide_device_ctx_sync(true).with_has_allocations(true);

  static void cpu_variant(legate::TaskContext context);
#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
  static void omp_variant(legate::TaskContext context);
#endif
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  static void gpu_variant(legate::TaskContext context);
#endif
};

class LEGATE_EXPORT FindBoundingBoxSorted : public LegateTask<FindBoundingBoxSorted> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp
    legate::TaskConfig{LocalTaskID{CoreTask::FIND_BOUNDING_BOX_SORTED}}.with_signature(
      legate::TaskSignature{}.inputs(1).outputs(1).scalars(0).redops(0).constraints(
        {Span<const legate::ProxyConstraint>{}})  // some compilers complain with {{}}
    );
  static constexpr VariantOptions GPU_VARIANT_OPTIONS =
    VariantOptions{}.with_elide_device_ctx_sync(true).with_has_allocations(true);

  static void cpu_variant(legate::TaskContext context);
#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
  static void omp_variant(legate::TaskContext context);
#endif
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  static void gpu_variant(legate::TaskContext context);
#endif
};

template <std::int32_t NDIM>
class ElementWiseMax {
 public:
  using LHS = Point<NDIM>;
  using RHS = Point<NDIM>;

  // Realm looks for a member of exactly this name
  static const Point<NDIM> identity;  // NOLINT(readability-identifier-naming)

  template <bool EXCLUSIVE>
  LEGATE_HOST_DEVICE inline static void apply(LHS& lhs, RHS rhs);
  template <bool EXCLUSIVE>
  LEGATE_HOST_DEVICE inline static void fold(RHS& rhs1, RHS rhs2);
};

template <std::int32_t NDIM>
class ElementWiseMin {
 public:
  using LHS = Point<NDIM>;
  using RHS = Point<NDIM>;

  // Realm looks for a member of exactly this name
  static const Point<NDIM> identity;  // NOLINT(readability-identifier-naming)

  template <bool EXCLUSIVE>
  LEGATE_HOST_DEVICE inline static void apply(LHS& lhs, RHS rhs);
  template <bool EXCLUSIVE>
  LEGATE_HOST_DEVICE inline static void fold(RHS& rhs1, RHS rhs2);
};

void register_partitioning_tasks(Library& core_lib);

}  // namespace legate::detail

#include <legate/partitioning/detail/partitioning_tasks.inl>
