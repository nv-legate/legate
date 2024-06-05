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

#include "core/partitioning/detail/partitioning_tasks.h"
#include "core/task/task_context.h"
#include "core/utilities/detail/omp_thread_local_storage.h"
#include "core/utilities/detail/unravel.h"

#include <omp.h>

namespace legate::detail {

namespace {

template <bool RECT>
struct FindBoundingBoxFn {
  template <std::int32_t POINT_NDIM, std::int32_t STORE_NDIM>
  void operator()(const legate::PhysicalStore& input, const legate::PhysicalStore& output)
  {
    auto out_acc       = output.write_accessor<Domain, 1>();
    const auto unravel = Unravel<STORE_NDIM>{input.shape<STORE_NDIM>()};
    auto result =
      Rect<POINT_NDIM>{ElementWiseMin<POINT_NDIM>::identity, ElementWiseMax<POINT_NDIM>::identity};

    if (unravel.empty()) {
      out_acc[0] = result;
      return;
    }

    const std::uint32_t num_threads = omp_get_max_threads();
    auto low                        = OMPThreadLocalStorage<Point<POINT_NDIM>>{num_threads};
    auto high                       = OMPThreadLocalStorage<Point<POINT_NDIM>>{num_threads};

    for (std::uint32_t tid = 0; tid < num_threads; ++tid) {
      low[tid]  = ElementWiseMin<POINT_NDIM>::identity;
      high[tid] = ElementWiseMax<POINT_NDIM>::identity;
    }

    if constexpr (RECT) {
      auto in_acc = input.read_accessor<Rect<POINT_NDIM>, STORE_NDIM>();
#pragma omp parallel
      {
        const std::uint32_t tid = omp_get_thread_num();
#pragma omp for schedule(static)
        for (std::uint64_t idx = 0; idx < unravel.volume(); ++idx) {
          const auto& rect = in_acc[unravel(idx)];
          if (rect.empty()) {
            continue;
          }
          ElementWiseMin<POINT_NDIM>::template apply<true>(low[tid], rect.lo);
          ElementWiseMax<POINT_NDIM>::template apply<true>(high[tid], rect.hi);
        }
      }
    } else {
      auto in_acc = input.read_accessor<Point<POINT_NDIM>, STORE_NDIM>();
#pragma omp parallel
      {
        const std::uint32_t tid = omp_get_thread_num();
#pragma omp for schedule(static)
        for (std::uint64_t idx = 0; idx < unravel.volume(); ++idx) {
          const auto& point = in_acc[unravel(idx)];
          ElementWiseMin<POINT_NDIM>::template apply<true>(low[tid], point);
          ElementWiseMax<POINT_NDIM>::template apply<true>(high[tid], point);
        }
      }
    }

    for (std::uint32_t tid = 0; tid < num_threads; ++tid) {
      ElementWiseMin<POINT_NDIM>::template apply<true>(result.lo, low[tid]);
      ElementWiseMax<POINT_NDIM>::template apply<true>(result.hi, high[tid]);
    }

    out_acc[0] = result;
  }
};

}  // namespace

/*static*/ void FindBoundingBox::omp_variant(legate::TaskContext context)
{
  auto input  = context.input(0).data();
  auto output = context.output(0).data();

  auto type = input.type();

  if (legate::is_rect_type(type)) {
    legate::double_dispatch(
      legate::ndim_rect_type(type), input.dim(), FindBoundingBoxFn<true>{}, input, output);
  } else {
    legate::double_dispatch(
      legate::ndim_point_type(type), input.dim(), FindBoundingBoxFn<false>{}, input, output);
  }
}

/*static*/ void FindBoundingBoxSorted::omp_variant(legate::TaskContext context)
{
  // The sorted case won't have any parallelism, so it's sufficient to call the cpu variant
  cpu_variant(context);
}

}  // namespace legate::detail
