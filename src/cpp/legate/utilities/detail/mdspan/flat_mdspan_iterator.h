/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/std/mdspan>

#include <cstddef>
#include <iterator>
#include <type_traits>

namespace legate::detail {

template <typename MDSpan>
class FlatMDSpanIterator;

template <typename El, typename Ex, typename L, typename A>
[[nodiscard]] constexpr
  typename FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>::difference_type
  operator-(const FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>& self,
            const FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>& other) noexcept;

template <typename El, typename Ex, typename L, typename A>
[[nodiscard]] constexpr FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>> operator-(
  FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>> self,
  typename FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>::difference_type n) noexcept;

template <typename El, typename Ex, typename L, typename A>
[[nodiscard]] constexpr FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>> operator+(
  FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>> self,
  typename FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>::difference_type n) noexcept;

template <typename El, typename Ex, typename L, typename A>
[[nodiscard]] constexpr FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>> operator+(
  typename FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>::difference_type n,
  FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>> self) noexcept;

template <typename El, typename Ex, typename L, typename A>
[[nodiscard]] constexpr bool operator==(
  const FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>& lhs,
  const FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>& rhs) noexcept;

template <typename El, typename Ex, typename L, typename A>
[[nodiscard]] constexpr bool operator!=(
  const FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>& lhs,
  const FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>& rhs) noexcept;

template <typename El, typename Ex, typename L, typename A>
[[nodiscard]] constexpr bool operator<(
  const FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>& lhs,
  const FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>& rhs) noexcept;

template <typename El, typename Ex, typename L, typename A>
[[nodiscard]] constexpr bool operator>(
  const FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>& lhs,
  const FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>& rhs) noexcept;

template <typename El, typename Ex, typename L, typename A>
[[nodiscard]] constexpr bool operator<=(
  const FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>& lhs,
  const FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>& rhs) noexcept;

template <typename El, typename Ex, typename L, typename A>
[[nodiscard]] constexpr bool operator>=(
  const FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>& lhs,
  const FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>& rhs) noexcept;

// ==========================================================================================

template <typename MDSpan>
class FlatMDSpanView;

/**
 * @brief An iterator over an `mdspan` that presents a flat view and allows
 * random elementwise access. It is particularly handy for passing to Thrust
 * algorithms to perform elementwise operations in parallel.
 */
template <typename Element, typename Extent, typename Layout, typename Accessor>
class FlatMDSpanIterator<::cuda::std::mdspan<Element, Extent, Layout, Accessor>> {
  class PointerWrapper {
   public:
    std::remove_const_t<Element> elem_;

    [[nodiscard]] constexpr Element* operator->() const noexcept;
  };

 public:
  using mdspan_type = ::cuda::std::mdspan<Element, Extent, Layout, Accessor>;
  using index_type  = typename mdspan_type::index_type;
  // The CCCL mdspan impls will fail to compile for 0-D, but in classic C++ fashion the
  // compiler error messages are inscrutable. So better to just guard against that mess with an
  // early static_assert()
  static_assert(mdspan_type::rank() >= 1, "Flat views over 0-D mdspans are not supported");

  using value_type        = std::remove_const_t<Element>;
  using reference         = typename mdspan_type::reference;
  using difference_type   = std::ptrdiff_t;
  using iterator_category = std::random_access_iterator_tag;
  using pointer           = std::conditional_t<std::is_lvalue_reference_v<reference>,
                                               std::add_pointer_t<reference>,
                                               PointerWrapper>;

  class ConstructKey {
    friend FlatMDSpanIterator;
    friend FlatMDSpanView<mdspan_type>;
  };

  /**
   * @brief Construct a flat mdspan iterator.
   *
   * @param span The span to view.
   * @param idx The linear index of the iterator (`0` for begin, `span.size()` for end).
   */
  constexpr explicit FlatMDSpanIterator(ConstructKey,
                                        const mdspan_type& span,
                                        index_type idx) noexcept;

  /**
   * @return A reference to the selected element.
   */
  [[nodiscard]] constexpr reference operator*() const noexcept;

  /**
   * @return A pointer to the selected element.
   */
  [[nodiscard]] constexpr pointer operator->() const noexcept;

  /**
   * @brief Pre-increment the iterator.
   *
   * @return A reference to this.
   */
  constexpr FlatMDSpanIterator& operator++() noexcept;

  /**
   * @brief Post-increment the iterator.
   *
   * @return A copy of the old iterator.
   */
  constexpr FlatMDSpanIterator operator++(int) noexcept;

  /**
   * @brief Pre-decrement the iterator.
   *
   * @return A reference to this.
   */
  constexpr FlatMDSpanIterator& operator--() noexcept;

  /**
   * @brief Post-decrement the iterator.
   *
   * @return A copy of the old iterator.
   */
  constexpr FlatMDSpanIterator operator--(int) noexcept;

  /**
   * @brief In-place add to the iterator.
   *
   * @param n The amount to increment.
   *
   * @return A reference to this.
   */
  constexpr FlatMDSpanIterator& operator+=(difference_type n) noexcept;

  /**
   * @brief In-place minus to the iterator.
   *
   * @param n The amount to decrement.
   *
   * @return A reference to this.
   */
  constexpr FlatMDSpanIterator& operator-=(difference_type n) noexcept;

  /**
   * @brief Access the iterator at a linear offset.
   *
   * @param n The linear index.
   *
   * @return A reference to the element at that index.
   */
  [[nodiscard]] constexpr reference operator[](difference_type n) const noexcept;

  friend difference_type operator-
    <>(const FlatMDSpanIterator& self, const FlatMDSpanIterator& other) noexcept;
  friend FlatMDSpanIterator operator- <>(FlatMDSpanIterator self, difference_type n) noexcept;

  friend FlatMDSpanIterator operator+ <>(FlatMDSpanIterator self, difference_type n) noexcept;
  friend FlatMDSpanIterator operator+ <>(difference_type n, FlatMDSpanIterator self) noexcept;

  friend bool operator== <>(const FlatMDSpanIterator& lhs, const FlatMDSpanIterator& rhs) noexcept;
  friend bool operator!= <>(const FlatMDSpanIterator& lhs, const FlatMDSpanIterator& rhs) noexcept;

  friend bool operator< <>(const FlatMDSpanIterator& lhs, const FlatMDSpanIterator& rhs) noexcept;
  friend bool operator><>(const FlatMDSpanIterator& lhs, const FlatMDSpanIterator& rhs) noexcept;

  friend bool operator<= <>(const FlatMDSpanIterator& lhs, const FlatMDSpanIterator& rhs) noexcept;
  friend bool operator>= <>(const FlatMDSpanIterator& lhs, const FlatMDSpanIterator& rhs) noexcept;

 private:
  const mdspan_type* span_{};
  index_type idx_{};
};

}  // namespace legate::detail

#include <legate/utilities/detail/mdspan/flat_mdspan_iterator.inl>
