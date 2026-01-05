/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/std/mdspan>

#include <cstddef>

namespace legate::detail {

/**
 * @brief A utility functor that generates a set of nested loops
 *
 * Invoking the `loop()` method of this class is roughly equivalent to
 *
 * @code{.cpp}
 * for (std::size_t i = 0; i < extents_type::extent(0); ++i) {
 *   for (std::size_t j = 0; j < extents_type::extent(1); ++j) {
 *     // ...
 *     fn(i, j, ...);
 *   }
 * }
 * @endcode
 *
 * Where the number of nested loops generated are equal to the rank of the extent type.
 *
 * @tparam ExtentsType The extent type to iterate over. Usually a `std::extents<...>`
 */
template <typename ExtentsType>
class NestedLooper {
 public:
  using extents_type = ExtentsType;
  using index_type   = typename extents_type::index_type;

  /**
   * @brief Execute a loop over the given extents.
   *
   * @param extents The extents to loop over.
   * @param fn The function to execute for each `i, j, k, ...`-th index of the loop.
   */
  template <typename F>
  static void loop(const extents_type& extents, F&& fn);

 private:
  /**
   * @brief Perform the actual loop.
   *
   * `indices` are not modified, and each recursive call of this function appends its index to
   * it, until they are all eventually passed to `fn`.
   *
   * @param extents The extents to loop over.
   * @param fn The function to execute.
   * @param indices The variadic pack of indices to (eventually) pass to `fn`.
   */
  template <std::size_t DIM, typename F, typename... Indices>
  static void loop_dispatch_(const extents_type& extents, F&& fn, Indices... indices);

  template <std::size_t DIM, typename F, typename... Indices, index_type... Is>
  static void static_loop_(const extents_type& extents,
                           F&& fn,
                           Indices... indices,
                           std::integer_sequence<index_type, Is...>);

  template <std::size_t DIM, typename F, typename... Indices>
  static void dynamic_loop_(const extents_type& extents, F&& fn, Indices... indices);
};

}  // namespace legate::detail

#include <legate/utilities/detail/mdspan/for_each_in_extent.inl>
