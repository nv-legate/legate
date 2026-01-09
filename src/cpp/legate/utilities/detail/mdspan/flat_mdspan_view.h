/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/mdspan/flat_mdspan_iterator.h>

#include <cuda/std/mdspan>

namespace legate::detail {

template <typename MDSpan>
class FlatMDSpanView;

/**
 * @brief A flattened view of an `mdspan` that allows efficient random
 * elementwise access.
 */
template <typename Element, typename Extent, typename Layout, typename Accessor>
class FlatMDSpanView<::cuda::std::mdspan<Element, Extent, Layout, Accessor>> {
 public:
  using mdspan_type    = ::cuda::std::mdspan<Element, Extent, Layout, Accessor>;
  using iterator       = FlatMDSpanIterator<mdspan_type>;
  using const_iterator = iterator;

  /**
   * @brief Construct a flat mdspan view.
   *
   * @param span The span to view.
   */
  constexpr explicit FlatMDSpanView(mdspan_type span) noexcept;

  /**
   * @return An iterator to the beginning of the range.
   */
  [[nodiscard]] constexpr iterator begin() const noexcept;

  /**
   * @return An iterator to the beginning of the range.
   */
  [[nodiscard]] constexpr iterator cbegin() const noexcept;

  /**
   * @return An iterator to the end of the range.
   */
  [[nodiscard]] constexpr iterator end() const noexcept;

  /**
   * @return An iterator to the beginning of the range.
   */
  [[nodiscard]] constexpr iterator cend() const noexcept;

 private:
  mdspan_type span_{};
};

template <typename T>
FlatMDSpanView(T span) -> FlatMDSpanView<T>;

}  // namespace legate::detail

#include <legate/utilities/detail/mdspan/flat_mdspan_view.inl>
