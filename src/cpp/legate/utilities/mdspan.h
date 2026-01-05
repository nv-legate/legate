/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/doxygen.h>
#include <legate/utilities/detail/mdspan/flat_mdspan_view.h>
#include <legate/utilities/typedefs.h>

#include <cuda/std/mdspan>

#include <cstddef>
#include <cstdint>

/**
 * @file
 */

namespace legate {

/**
 * @addtogroup util
 * @{
 */

/**
 * @brief Create a flattened view of an `mdspan` that allows efficient random
 * elementwise access.
 *
 * The returned view object supports all the usual iterator semantics.
 *
 * Unfortunately, flattening mdspan into a linear iterator ends up with inefficient code-gen as
 * compilers are unable to untangle the internal state required to make this work. This is not
 * really an "implementation quality" issue so much as a fundamental constraint. In order to
 * implement iterators, you need to solve the problem of mapping a linear index to a
 * N-dimensional point in space. This linearization is done via the following:
 *
 * @code{.cpp}
 * std::array<std::size_t, DIM> point;
 *
 * for (auto dim = DIM; dim-- > 0;) {
 *   point[dim] = index % span.extent(dim);
 *   index /= span.extent(dim);
 * }
 * @endcode
 *
 * The problem are the modulus and div commands. Modern compilers are seemingly unable to hoist
 * those computations out of the loop and vectorize the code. So an equivalent loop over the
 * extents "normally":
 *
 * @code{.cpp}
 * for (std::size_t i = 0; i < span.extent(0); ++i) {
 *   for (std::size_t j = 0; j < span.extent(1); ++j) {
 *     span(i, j) = ...
 *   }
 * }
 * @endcode
 *
 * Will be fully vectorized by optimizers, but the following (which is more or less what
 * this iterator expands to):
 *
 * @code{.cpp}
 * for (std::size_t i = 0; i < PROD(span.extents()...); ++i) {
 *   std::array<std::size_t, DIM> point = delinearize(i);
 *
 *   span(point) = ...
 * }
 * @endcode
 *
 * Defeats all known modern optimizing compilers. Therefore, unless this iterator is truly
 * required, the user is **strongly** encouraged to iterate over their mdspan normally.
 *
 * @param span The mdspan to flatten.
 *
 * @return The flat view.
 */
template <typename Element, typename Extent, typename Layout, typename Accessor>
[[nodiscard]] constexpr detail::FlatMDSpanView<
  ::cuda::std::mdspan<Element, Extent, Layout, Accessor>>
flatten(::cuda::std::mdspan<Element, Extent, Layout, Accessor> span) noexcept;

// ==========================================================================================

/**
 * @brief Execute a function `fn` for each `i, j, k, ...`-th point in an extent `extents`.
 *
 * Invoking this method is roughly equivalent to
 *
 * @code{.cpp}
 * for (std::size_t i = 0; i < extents.extent(0); ++i) {
 *   for (std::size_t j = 0; j < extents.extent(1); ++j) {
 *     // ...
 *     fn(i, j, ...);
 *   }
 * }
 * @endcode
 *
 * Where the number of nested loops generated are equal to the rank of the extent.
 *
 * The utility of this function is multi-fold:
 *
 * #. It allow efficient iteration over an mdspan of variable dimension.
 * #. It separates the iteration from the container. For example, if the user wanted to iterate
 *    over the intersection of multiple mdspans, they could compute the intersection of their
 *    extents, and use this function to generate the loops.
 *
 * @param extents The extents to iterate over.
 * @param fn The function to execute.
 */
template <typename IndexType, std::size_t... Extents, typename F>
void for_each_in_extent(const ::cuda::std::extents<IndexType, Extents...>& extents, F&& fn);

/**
 * @brief Execute a function `fn` for each `i, j, k, ...`-th index in point `point`.
 *
 * This routine treats `point` as an "extent", where each index of `point` gives the 0-based
 * extent for that dimension. So given a 2D point `<1, 1>`, then this routine would generate
 * the following calls:
 *
 * - `fn(0, 0)`
 * - `fn(0, 1)`
 * - `fn(1, 0)`
 * - `fn(1, 1)`
 *
 * @param point The `Point` to iterate over.
 * @param fn The function to execute.
 */
template <std::int32_t DIM, typename F>
void for_each_in_extent(const Point<DIM>& point, F&& fn);

/**
 * @brief Execute a function `fn` for each `i, j, k, ...`-th index in rect `rect`.
 *
 * This routine is similar to the `Point` overload, except that the extents are given by the
 * difference between `rect[i].lo` and `rect[i].hi`. The indices are then converted to 0-based
 * indices before being passed to `fn`. So given a 2D rect: `[<1, 1>, <2, 2>]`, then this
 * routine would generate the following calls:
 *
 * - `fn(0, 0)`
 * - `fn(0, 1)`
 * - `fn(0, 2)`
 * - `fn(1, 0)`
 * - `fn(1, 1)`
 * - `fn(1, 2)`
 * - `fn(2, 0)`
 * - `fn(2, 1)`
 * - `fn(2, 2)`
 *
 * @param rect The `Rect` to iterate over.
 * @param fn The function to execute.
 */
template <std::int32_t DIM, typename F>
void for_each_in_extent(const Rect<DIM>& rect, F&& fn);

/** @} */

}  // namespace legate

#include <legate/utilities/mdspan.inl>
