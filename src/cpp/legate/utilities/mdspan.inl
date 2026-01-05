/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/mdspan/for_each_in_extent.h>
#include <legate/utilities/detail/mdspan/util.h>
#include <legate/utilities/mdspan.h>

namespace legate {

template <typename El, typename Ex, typename L, typename A>
constexpr detail::FlatMDSpanView<::cuda::std::mdspan<El, Ex, L, A>> flatten(
  ::cuda::std::mdspan<El, Ex, L, A> span) noexcept
{
  return detail::FlatMDSpanView<::cuda::std::mdspan<El, Ex, L, A>>{std::move(span)};
}

// ==========================================================================================

template <typename IndexType, std::size_t... Extents, typename F>
void for_each_in_extent(const ::cuda::std::extents<IndexType, Extents...>& extents, F&& fn)
{
  detail::NestedLooper<::cuda::std::extents<IndexType, Extents...>>::loop(extents,
                                                                          std::forward<F>(fn));
}

template <std::int32_t DIM, typename F>
void for_each_in_extent(const Point<DIM>& point, F&& fn)
{
  for_each_in_extent(detail::mdspan_detail::dynamic_extents(point), std::forward<F>(fn));
}

template <std::int32_t DIM, typename F>
void for_each_in_extent(const Rect<DIM>& rect, F&& fn)
{
  for_each_in_extent(detail::mdspan_detail::dynamic_extents(rect), std::forward<F>(fn));
}

}  // namespace legate
