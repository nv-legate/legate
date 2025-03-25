/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/doxygen.h>

#include <cstddef>
#include <iterator>
#include <type_traits>

/**
 * @file
 * @brief Class definition for legate::Span
 */

namespace legate {

namespace detail {

template <typename T, typename = void>
struct is_container : std::false_type {};

template <typename T>
struct is_container<
  T,
  std::void_t<decltype(std::data(std::declval<T>()), std::size(std::declval<T>()))>>
  : std::true_type {};

template <typename T>
inline constexpr bool is_container_v = is_container<T>::value;

}  // namespace detail

/**
 * @addtogroup data
 * @{
 */

/**
 * @brief A simple span implementation used in Legate.
 *
 * Should eventually be replaced with std::span once we bump up the C++ standard version to C++20
 */
template <typename T>
class Span {
 public:
  using element_type           = T;
  using value_type             = std::remove_cv_t<T>;
  using size_type              = std::size_t;
  using difference_type        = std::ptrdiff_t;
  using pointer                = T*;
  using const_pointer          = const T*;
  using reference              = T&;
  using const_reference        = const T&;
  using iterator               = pointer;
  using const_iterator         = const_pointer;
  using reverse_iterator       = std::reverse_iterator<iterator>;
  using reverse_const_iterator = std::reverse_iterator<const_iterator>;

  constexpr Span() = default;

  /**
   * @brief Construct a span from a container-like object.
   *
   * This overload only participates in overload resolution if C satisfies _ContainerLike_. It
   * must have a valid overload of `std::data()` and `std::size()` which refer to a contiguous
   * buffer of data and its size respectively.
   *
   * @param container The container-like object.
   */
  template <typename C,
            typename = std::enable_if_t<detail::is_container_v<C> &&
                                        !std::is_same_v<C, std::initializer_list<T>>>>
  constexpr Span(C& container);  // NOLINT(google-explicit-constructor)

  /**
   * @brief Construct a span from an initializer list of items directly.
   *
   * This overload is relatively dangerous insofar that the span can very easily outlive the
   * initializer list. It is generally only preferred to target this overload when taking a
   * `Span` as a function argument where the ability to simply do `foo({1, 2, 3, 4})` is
   * preferred.
   *
   * @param il The initializer list.
   */
  constexpr Span(std::initializer_list<T> il);

  template <typename It>
  constexpr Span(It begin, It end);

  /**
   * @brief Creates a span with an existing pointer and a size.
   *
   * The caller must guarantee that the allocation is big enough (i.e., bigger than or
   * equal to `sizeof(T) * size`) and that the allocation is alive while the span is alive.
   *
   * @param data Pointer to the data
   * @param size Number of elements
   */
  constexpr Span(T* data, size_type size);

  /**
   * @brief Returns the number of elements
   *
   * @return The number of elements
   */
  [[nodiscard]] constexpr size_type size() const;

  [[nodiscard]] constexpr reference operator[](size_type pos) const;

  /**
   * @brief Returns the pointer to the first element
   *
   * @return Pointer to the first element
   */
  [[nodiscard]] constexpr const_iterator begin() const;

  /**
   * @brief Returns the pointer to the end of allocation
   *
   * @return Pointer to the end of allocation
   */
  [[nodiscard]] constexpr const_iterator end() const;

  /**
   * @brief Returns the pointer to the first element
   *
   * @return Pointer to the first element
   */
  [[nodiscard]] constexpr iterator begin();

  /**
   * @brief Returns the pointer to the end of allocation
   *
   * @return Pointer to the end of allocation
   */
  [[nodiscard]] constexpr iterator end();

  /**
   * @return An iterator to the last element.
   */
  [[nodiscard]] constexpr reverse_const_iterator rbegin() const;

  /**
   * @return An iterator one past the first element.
   */
  [[nodiscard]] constexpr reverse_const_iterator rend() const;

  /**
   * @return An iterator to the last element.
   */
  [[nodiscard]] constexpr reverse_iterator rbegin();

  /**
   * @return An iterator one past the first element.
   */
  [[nodiscard]] constexpr reverse_iterator rend();

  /**
   * @return A reference to the first element in the span.
   */
  [[nodiscard]] constexpr reference front() const;

  /**
   * @return A reference to the last element in the span.
   */
  [[nodiscard]] constexpr reference back() const;

  /**
   * @brief Slices off the first `off` elements. Passing an `off` greater than
   * the size will fail with an assertion failure.
   *
   * @param off Number of elements to skip
   *
   * @return A span for range `[off, size())`
   */
  [[nodiscard]] constexpr Span subspan(size_type off);

  /**
   * @brief Returns a `const` pointer to the data
   *
   * @return Pointer to the data
   */
  [[nodiscard]] constexpr const_pointer ptr() const;

  /**
   * @brief Returns a pointer to the data.
   *
   * @return Pointer to the data.
   */
  [[nodiscard]] constexpr pointer data() const;

 private:
  struct container_tag {};

  template <typename C>
  constexpr Span(C& container, container_tag);

  struct size_tag {};

  constexpr Span(T* data, std::size_t size, size_tag);

  T* data_{};
  std::size_t size_{};
};

/** @} */

}  // namespace legate

#include <legate/utilities/span.inl>
