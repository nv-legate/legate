/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/partitioning_tasks.h>

#include <legate/task/task_context.h>
#include <legate/utilities/dispatch.h>

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

LEGION_FOREACH_N(IDENTITIES)

#undef IDENTITIES

namespace {

template <bool RECT>
class FindBoundingBoxFn {
 public:
  template <std::int32_t POINT_NDIM, std::int32_t STORE_NDIM>
  void operator()(const legate::PhysicalStore& input, legate::PhysicalStore& output) const
  {
    auto result =
      Rect<POINT_NDIM>{ElementWiseMin<POINT_NDIM>::identity, ElementWiseMax<POINT_NDIM>::identity};

    if constexpr (RECT) {
      const auto in_acc = input.span_read_accessor<Rect<POINT_NDIM>, STORE_NDIM>();

      legate::for_each_in_extent(in_acc.extents(), [&](auto... Is) {
        if (auto&& rect = in_acc(Is...); !rect.empty()) {
          ElementWiseMin<POINT_NDIM>::template apply<true>(result.lo, rect.lo);
          ElementWiseMax<POINT_NDIM>::template apply<true>(result.hi, rect.hi);
        }
      });
    } else {
      const auto in_acc = input.span_read_accessor<Point<POINT_NDIM>, STORE_NDIM>();

      legate::for_each_in_extent(in_acc.extents(), [&](auto... Is) {
        auto&& point = in_acc(Is...);

        ElementWiseMin<POINT_NDIM>::template apply<true>(result.lo, point);
        ElementWiseMax<POINT_NDIM>::template apply<true>(result.hi, point);
      });
    }
    output.span_write_accessor<Domain, 1>()[0] = result;
  }
};

template <bool RECT>
class FindBoundingBoxSortedFn {
 public:
  template <std::int32_t POINT_NDIM, std::int32_t STORE_NDIM>
  void operator()(const legate::PhysicalStore& input, legate::PhysicalStore& output) const
  {
    const auto out_acc = output.span_write_accessor<Domain, 1>();
    const auto shape   = input.shape<STORE_NDIM>();

    if (shape.empty()) {
      out_acc[0] = Rect<POINT_NDIM>{ElementWiseMin<POINT_NDIM>::identity,
                                    ElementWiseMax<POINT_NDIM>::identity};
      return;
    }

    Rect<POINT_NDIM> result;
    if constexpr (RECT) {
      const auto in_acc     = input.span_read_accessor<Rect<POINT_NDIM>, STORE_NDIM>();
      const auto flat_view  = legate::flatten(in_acc);
      const auto& last_rect = *std::prev(flat_view.end());
      result                = *flat_view.begin();

      ElementWiseMin<POINT_NDIM>::template apply<true>(result.lo, last_rect.lo);
      ElementWiseMax<POINT_NDIM>::template apply<true>(result.hi, last_rect.hi);
    } else {
      const auto in_acc       = input.span_read_accessor<Point<POINT_NDIM>, STORE_NDIM>();
      const auto flat_view    = legate::flatten(in_acc);
      const auto& first_point = *flat_view.begin();
      const auto& last_point  = *std::prev(flat_view.end());

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

void register_partitioning_tasks(Library& core_lib)
{
  FindBoundingBox::register_variants(legate::Library{&core_lib});
  FindBoundingBoxSorted::register_variants(legate::Library{&core_lib});
}

}  // namespace legate::detail
