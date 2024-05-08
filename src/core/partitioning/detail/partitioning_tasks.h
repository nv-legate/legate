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

#include "core/task/task.h"
#include "core/utilities/typedefs.h"

namespace legate::detail {

class Library;

class FindBoundingBox : public LegateTask<FindBoundingBox> {
 public:
  static void cpu_variant(legate::TaskContext context);
#if LegateDefined(LEGATE_USE_OPENMP)
  static void omp_variant(legate::TaskContext context);
#endif
#if LegateDefined(LEGATE_USE_CUDA)
  static void gpu_variant(legate::TaskContext context);
#endif
};

class FindBoundingBoxSorted : public LegateTask<FindBoundingBoxSorted> {
 public:
  static void cpu_variant(legate::TaskContext context);
#if LegateDefined(LEGATE_USE_OPENMP)
  static void omp_variant(legate::TaskContext context);
#endif
#if LegateDefined(LEGATE_USE_CUDA)
  static void gpu_variant(legate::TaskContext context);
#endif
};

template <std::int32_t NDIM>
class ElementWiseMax {
 public:
  using LHS = Point<NDIM>;
  using RHS = Point<NDIM>;

  static const Point<NDIM> identity;

  template <bool EXCLUSIVE>
  __CUDA_HD__ inline static void apply(LHS& lhs, RHS rhs);
  template <bool EXCLUSIVE>
  __CUDA_HD__ inline static void fold(RHS& rhs1, RHS rhs2);
};

template <std::int32_t NDIM>
class ElementWiseMin {
 public:
  using LHS = Point<NDIM>;
  using RHS = Point<NDIM>;

  static const Point<NDIM> identity;

  template <bool EXCLUSIVE>
  __CUDA_HD__ inline static void apply(LHS& lhs, RHS rhs);
  template <bool EXCLUSIVE>
  __CUDA_HD__ inline static void fold(RHS& rhs1, RHS rhs2);
};

void register_partitioning_tasks(Library* core_lib);

}  // namespace legate::detail

#include "core/partitioning/detail/partitioning_tasks.inl"
