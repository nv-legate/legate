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

#include "core/partitioning/detail/partitioning_tasks.h"

#include "core/legate_c.h"
#include "core/task/task_context.h"
#include "core/utilities/dispatch.h"

#include <limits>

namespace legate::detail {

namespace {

template <std::int32_t N>
Point<N> create_point(coord_t val) noexcept
{
  try {
    return Point<N>{val};
  } catch (...) {
    LEGATE_ABORT("Something bad happened");
  }
}

}  // namespace

#define IDENTITIES(N)                                               \
  template <>                                                       \
  const Point<N> ElementWiseMax<N>::identity{                       \
    create_point<N>(legate::MaxReduction<std::int64_t>::identity)}; \
  template <>                                                       \
  const Point<N> ElementWiseMin<N>::identity{                       \
    create_point<N>(legate::MinReduction<std::int64_t>::identity)};

#if LEGATE_MAX_DIM >= 1
IDENTITIES(1)
#endif
#if LEGATE_MAX_DIM >= 2
IDENTITIES(2)
#endif
#if LEGATE_MAX_DIM >= 3
IDENTITIES(3)
#endif
#if LEGATE_MAX_DIM >= 4
IDENTITIES(4)
#endif
#if LEGATE_MAX_DIM >= 5
IDENTITIES(5)
#endif
#if LEGATE_MAX_DIM >= 6
IDENTITIES(6)
#endif
#if LEGATE_MAX_DIM >= 7
IDENTITIES(7)
#endif
#if LEGATE_MAX_DIM >= 8
IDENTITIES(8)
#endif
#if LEGATE_MAX_DIM >= 9
IDENTITIES(9)
#endif

#undef IDENTITIES

namespace {

template <bool RECT>
struct FindBoundingBoxFn {
  template <std::int32_t POINT_NDIM, std::int32_t STORE_NDIM>
  void operator()(const legate::PhysicalStore& input, const legate::PhysicalStore& output)
  {
    auto out_acc = output.write_accessor<Domain, 1>();
    auto shape   = input.shape<STORE_NDIM>();
    auto result =
      Rect<POINT_NDIM>{ElementWiseMin<POINT_NDIM>::identity, ElementWiseMax<POINT_NDIM>::identity};

    if (shape.empty()) {
      out_acc[0] = result;
      return;
    }

    if constexpr (RECT) {
      auto in_acc = input.read_accessor<Rect<POINT_NDIM>, STORE_NDIM>(shape);
      for (PointInRectIterator<STORE_NDIM> it{shape}; it.valid(); ++it) {
        const auto& rect = in_acc[*it];
        if (rect.empty()) {
          continue;
        }
        ElementWiseMin<POINT_NDIM>::template apply<true>(result.lo, rect.lo);
        ElementWiseMax<POINT_NDIM>::template apply<true>(result.hi, rect.hi);
      }
    } else {
      auto in_acc = input.read_accessor<Point<POINT_NDIM>, STORE_NDIM>(shape);
      for (PointInRectIterator<STORE_NDIM> it{shape}; it.valid(); ++it) {
        const auto& point = in_acc[*it];
        ElementWiseMin<POINT_NDIM>::template apply<true>(result.lo, point);
        ElementWiseMax<POINT_NDIM>::template apply<true>(result.hi, point);
      }
    }

    out_acc[0] = result;
  }
};

template <bool RECT>
struct FindBoundingBoxSortedFn {
  template <std::int32_t POINT_NDIM, std::int32_t STORE_NDIM>
  void operator()(const legate::PhysicalStore& input, const legate::PhysicalStore& output)
  {
    auto out_acc = output.write_accessor<Domain, 1>();
    auto shape   = input.shape<STORE_NDIM>();

    if (shape.empty()) {
      out_acc[0] = Rect<POINT_NDIM>{ElementWiseMin<POINT_NDIM>::identity,
                                    ElementWiseMax<POINT_NDIM>::identity};
      return;
    }

    Rect<POINT_NDIM> result;
    if constexpr (RECT) {
      auto in_acc           = input.read_accessor<Rect<POINT_NDIM>, STORE_NDIM>(shape);
      result                = in_acc[shape.lo];
      const auto& last_rect = in_acc[shape.hi];

      ElementWiseMin<POINT_NDIM>::template apply<true>(result.lo, last_rect.lo);
      ElementWiseMax<POINT_NDIM>::template apply<true>(result.hi, last_rect.hi);
    } else {
      auto in_acc             = input.read_accessor<Point<POINT_NDIM>, STORE_NDIM>(shape);
      const auto& first_point = in_acc[shape.lo];
      const auto& last_point  = in_acc[shape.hi];

      result = Rect<POINT_NDIM>{first_point, first_point};
      ElementWiseMin<POINT_NDIM>::template apply<true>(result.lo, last_point);
      ElementWiseMax<POINT_NDIM>::template apply<true>(result.hi, last_point);
    }

    out_acc[0] = result;
  }
};

}  // namespace

/*static*/ void FindBoundingBox::cpu_variant(legate::TaskContext context)
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

/*static*/ void FindBoundingBoxSorted::cpu_variant(legate::TaskContext context)
{
  auto input  = context.input(0).data();
  auto output = context.output(0).data();

  auto type = input.type();

  if (legate::is_rect_type(type)) {
    legate::double_dispatch(
      legate::ndim_rect_type(type), input.dim(), FindBoundingBoxSortedFn<true>{}, input, output);
  } else {
    legate::double_dispatch(
      legate::ndim_point_type(type), input.dim(), FindBoundingBoxSortedFn<false>{}, input, output);
  }
}

void register_partitioning_tasks(Library* core_lib)
{
  FindBoundingBox::register_variants(legate::Library{core_lib}, LEGATE_CORE_FIND_BOUNDING_BOX);
  FindBoundingBoxSorted::register_variants(legate::Library{core_lib},
                                           LEGATE_CORE_FIND_BOUNDING_BOX_SORTED);
}

}  // namespace legate::detail
