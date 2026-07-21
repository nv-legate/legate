/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/detail/linearize.h>

#include <legate/utilities/dispatch.h>

#include <cstddef>
#include <cstdint>

namespace legate::detail {

namespace {

class LinearizeFn {
 public:
  template <std::int32_t DIM>
  [[nodiscard]] coord_t operator()(const Point& lo, const Point& hi, const Point& point) const;
};

template <std::int32_t DIM>
coord_t LinearizeFn::operator()(const Point& lo, const Point& hi, const Point& point) const
{
  coord_t idx = 0;

  for (std::uint32_t dim = 0; dim < DIM; ++dim) {
    idx = (idx * (hi[dim] - lo[dim] + 1)) + point[dim] - lo[dim];
  }
  return idx;
}

}  // namespace

std::size_t linearize(const Point& lo, const Point& hi, const Point& point)
{
  if (point.dim() == 0) {
    return 0;
  }
  return static_cast<std::size_t>(dim_dispatch(point.dim(), LinearizeFn{}, lo, hi, point));
}

namespace {

class DelinearizeFn {
 public:
  template <std::int32_t DIM>
  [[nodiscard]] Point operator()(const Point& lo, const Point& hi, coord_t idx) const;
};

template <std::int32_t DIM>
Point DelinearizeFn::operator()(const Point& lo, const Point& hi, coord_t idx) const
{
  static_assert(DIM >= 1);

  auto point = Point::zeroes(DIM);

  for (std::int32_t dim = DIM - 1; dim >= 0; --dim) {
    const auto udim   = static_cast<std::uint32_t>(dim);
    const auto extent = hi[udim] - lo[udim] + 1;

    point[udim] = (idx % extent) + lo[udim];
    idx /= extent;
  }
  return point;
}

}  // namespace

Point delinearize(const Point& lo, const Point& hi, std::size_t idx)
{
  if (lo.dim() == 0) {
    return Point{};
  }
  return dim_dispatch(lo.dim(), DelinearizeFn{}, lo, hi, static_cast<coord_t>(idx));
}

}  // namespace legate::detail
