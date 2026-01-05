/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/mdspan/for_each_in_extent.h>

#include <cuda/std/mdspan>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace legate::detail {

template <typename E>
template <std::size_t DIM, typename F, typename... Indices>
/* static */ void NestedLooper<E>::loop_dispatch_(const extents_type& extents,
                                                  F&& fn,
                                                  Indices... indices)
{
  static_assert((... && std::is_same_v<Indices, index_type>));
  static_assert(DIM < extents_type::rank());

  constexpr auto EXT = extents_type::static_extent(DIM);

  if constexpr (EXT == ::cuda::std::dynamic_extent) {
    dynamic_loop_<DIM>(extents, std::forward<F>(fn), indices...);
  } else {
    static_loop_<DIM>(
      extents, std::forward<F>(fn), indices..., std::make_integer_sequence<index_type, EXT>{});
  }
}

template <typename E>
template <std::size_t DIM, typename F, typename... Indices, typename E::index_type... Is>
/* static */ void NestedLooper<E>::static_loop_(const extents_type& extents,
                                                F&& fn,
                                                Indices... indices,
                                                std::integer_sequence<index_type, Is...>)
{
  static_assert(extents_type::static_extent(DIM) == sizeof...(Is));
  if constexpr (DIM == extents_type::rank() - 1) {  // reached bottom
    (static_cast<void>(fn(indices..., Is)), ...);
  } else {
    (loop_dispatch_<DIM + 1>(
       extents,
       // We take fn by &&, so this is always OK (it will always be a reference)
       std::forward<F>(fn),  // NOLINT(bugprone-use-after-move)
       indices...,
       Is),
     ...);
  }
}

template <typename E>
template <std::size_t DIM, typename F, typename... Indices>
/* static */ void NestedLooper<E>::dynamic_loop_(const extents_type& extents,
                                                 F&& fn,
                                                 Indices... indices)
{
  const auto dim_ext = extents.extent(DIM);

  for (index_type i = 0; i < dim_ext; ++i) {
    if constexpr (DIM == extents_type::rank() - 1) {  // reached bottom
      static_cast<void>(fn(indices..., i));
    } else {
      loop_dispatch_<DIM + 1>(
        extents,
        // We take fn by &&, so this is always OK (it will always be a reference)
        std::forward<F>(fn),  // NOLINT(bugprone-use-after-move)
        indices...,
        i);
    }
  }
}

template <typename E>
template <typename F>
/* static */ void NestedLooper<E>::loop(const extents_type& extents, F&& fn)
{
  if constexpr (extents_type::rank() == 0) {
    return;
  } else {
    loop_dispatch_<0>(extents, std::forward<F>(fn));
  }
}

}  // namespace legate::detail
