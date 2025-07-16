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
 * @brief Class definition for legate::Span.
 */

namespace legate {

template <typename T>
class tuple;

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
 * Should eventually be replaced with std::span once we bump up the C++ standard version to C++20.
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
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

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
            typename = std::enable_if_t<
              detail::is_container_v<C> &&                     // NOLINT(modernize-type-traits)
              !std::is_same_v<C, std::initializer_list<T>> &&  // NOLINT(modernize-type-traits)
              !std::is_same_v<C, tuple<value_type>>>>          // NOLINT(modernize-type-traits)
  constexpr Span(C& container);  // NOLINT(google-explicit-constructor)

  /**
   * @brief Construct a `Span` from a `tuple`.
   *
   * This overload must exist because `tuple::data()` returns a reference to the underlying
   * container, not the underlying pointer (as is usually expected by `data()`) which causes
   * the `Span(C& container)` overload to brick.
   *
   * @param tup The tuple to construct the span from.
   */
  constexpr Span(const tuple<value_type>& tup);  // NOLINT(google-explicit-constructor)

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
   * @param data Pointer to the data.
   * @param size Number of elements.
   */
  constexpr Span(T* data, size_type size);

  /**
   * @brief Returns the number of elements.
   *
   * @return The number of elements.
   */
  [[nodiscard]] constexpr size_type size() const;

  /**
   * @return `true` if the span has size 0, `false` otherwise.
   */
  [[nodiscard]] constexpr bool empty() const;

  [[nodiscard]] constexpr reference operator[](size_type pos) const;

  /**
   * @brief Access an element with bounds checking.
   *
   * @param pos The index of the element to access.
   *
   * @return A reference to the element as index `pos`
   *
   * @throw std::out_of_range if `pos` is not in bounds of the span.
   */
  [[nodiscard]] constexpr reference at(size_type pos) const;

  /**
   * @brief Returns the pointer to the first element.
   *
   * @return Pointer to the first element.
   */
  [[nodiscard]] constexpr const_iterator cbegin() const noexcept;

  /**
   * @brief Returns the pointer to the end of allocation.
   *
   * @return Pointer to the end of allocation.
   */
  [[nodiscard]] constexpr const_iterator cend() const noexcept;
  /**
   * @brief Returns the pointer to the first element.
   *
   * @return Pointer to the first element.
   */
  [[nodiscard]] constexpr const_iterator begin() const;

  /**
   * @brief Returns the pointer to the end of allocation.
   *
   * @return Pointer to the end of allocation.
   */
  [[nodiscard]] constexpr const_iterator end() const;

  /**
   * @brief Returns the pointer to the first element.
   *
   * @return Pointer to the first element.
   */
  [[nodiscard]] constexpr iterator begin();

  /**
   * @brief Returns the pointer to the end of allocation.
   *
   * @return Pointer to the end of allocation.
   */
  [[nodiscard]] constexpr iterator end();

  /**
   * @return An iterator to the last element.
   */
  [[nodiscard]] constexpr const_reverse_iterator crbegin() const noexcept;

  /**
   * @return An iterator to the location preceding the first element.
   */
  [[nodiscard]] constexpr const_reverse_iterator crend() const noexcept;

  /**
   * @return An iterator to the last element.
   */
  [[nodiscard]] constexpr const_reverse_iterator rbegin() const;

  /**
   * @return An iterator to the location preceding the first element.
   */
  [[nodiscard]] constexpr const_reverse_iterator rend() const;

  /**
   * @return An iterator to the last element.
   */
  [[nodiscard]] constexpr reverse_iterator rbegin();

  /**
   * @return An iterator to the location preceding the first element.
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
   * @param off Number of elements to skip.
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

  /**
   * @brief Compare the values of the span for equality.
   *
   * Since span is fundamentally a "view" or "handle" type, when a user writes:
   * ```cpp
   * span == other_span
   * ```
   * It is not immediately clear what they mean to compare. Should `operator==()` behave like
   * `std::shared_ptr` and compare the pointer values? Should it compare the container values?
   * For this reason, the standard does not actually define `operator==()` (or any of the other
   * comparison operators) for `std::span`, and we don't either. Instead, we provide this
   * helper function to make the comparison explicit.
   *
   * If we ever have need for pointer comparisons, we could add `shallow_compare()`, but then
   * it's just as easy (and expressive) for the user to write
   * ```cpp
   * span.data() == other_span.data()
   * ```
   *
   * @param other The span to compare against.
   *
   * @return `true` if all values of this span are equal to that of `other`, `false` otherwise.
   */
  [[nodiscard]] constexpr bool deep_equal(const Span<const value_type>& other) const;

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
