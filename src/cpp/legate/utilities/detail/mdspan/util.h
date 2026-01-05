/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/typedefs.h>

#include <cuda/std/array>
#include <cuda/std/mdspan>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace legate::detail::mdspan_detail {

template <std::int32_t DIM, std::size_t... Is>
[[nodiscard]] ::cuda::std::array<coord_t, DIM> to_array(const Point<DIM>& point,
                                                        std::index_sequence<Is...>)
{
  static_assert(DIM == sizeof...(Is));
  return {point[Is]...};
}

template <std::int32_t DIM>
[[nodiscard]] ::cuda::std::array<coord_t, DIM> to_array(const Point<DIM>& point)
{
  return to_array(point, std::make_index_sequence<DIM>{});
}

template <std::int32_t DIM, std::size_t... Is>
[[nodiscard]] ::cuda::std::dextents<coord_t, DIM> dynamic_extents(const Point<DIM>& point,
                                                                  std::index_sequence<Is...>)
{
  static_assert(DIM == sizeof...(Is));
  return ::cuda::std::dextents<coord_t, DIM>{point[Is]...};
}

template <std::int32_t DIM>
[[nodiscard]] ::cuda::std::dextents<coord_t, DIM> dynamic_extents(const Point<DIM>& point)
{
  return dynamic_extents(point, std::make_index_sequence<DIM>{});
}

template <std::int32_t DIM, std::size_t... Is>
[[nodiscard]] ::cuda::std::dextents<coord_t, DIM> dynamic_extents(const Rect<DIM>& rect,
                                                                  std::index_sequence<Is...>)
{
  static_assert(DIM == sizeof...(Is));
  return ::cuda::std::dextents<coord_t, DIM>{
    // Handle empty dims by clamping to 0
    std::max(1 + rect.hi[Is] - rect.lo[Is], coord_t{0})...};
}

template <std::int32_t DIM>
[[nodiscard]] ::cuda::std::dextents<coord_t, DIM> dynamic_extents(const Rect<DIM>& rect)
{
  return dynamic_extents(rect, std::make_index_sequence<DIM>{});
}

}  // namespace legate::detail::mdspan_detail
