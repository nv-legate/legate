/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "legate/task/task.h"
#include "legate/task/variant_options.h"
#include "legate/utilities/detail/core_ids.h"
#include "legate/utilities/typedefs.h"

#include <cstdint>

namespace legate::detail {

class Library;

class FindBoundingBox : public LegateTask<FindBoundingBox> {
 public:
  static constexpr auto TASK_ID = LocalTaskID{CoreTask::FIND_BOUNDING_BOX};
  static constexpr VariantOptions GPU_VARIANT_OPTIONS =
    VariantOptions{}.with_elide_device_ctx_sync(true);

  static void cpu_variant(legate::TaskContext context);
#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
  static void omp_variant(legate::TaskContext context);
#endif
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  static void gpu_variant(legate::TaskContext context);
#endif
};

class FindBoundingBoxSorted : public LegateTask<FindBoundingBoxSorted> {
 public:
  static constexpr auto TASK_ID = LocalTaskID{CoreTask::FIND_BOUNDING_BOX_SORTED};
  static constexpr VariantOptions GPU_VARIANT_OPTIONS =
    VariantOptions{}.with_elide_device_ctx_sync(true);

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

void register_partitioning_tasks(Library* core_lib);

}  // namespace legate::detail

#include "legate/partitioning/detail/partitioning_tasks.inl"
