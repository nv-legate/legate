/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/span.h>

#include <cuda/std/inplace_vector>

#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <type_traits>
#include <variant>
#include <vector>

namespace legate::detail {

template <typename T, std::uint32_t>
class SmallVector;

namespace small_vector_detail {

template <typename T>
[[nodiscard]] constexpr std::uint32_t storage_size()
{
  // Like LLVM's SmallVector, the point of our SmallVector is that it holds at least one value
  // on the stack. But if the type is arbitrarily large, then this also ends up allocating
  // arbitrarily large amounts of stack storage, which is not good.
  //
  // If this is indeed what the user wants, they should need to opt in to it explicitly. 256 is
  // a bit of a magic number, but it's the same as LLVM uses. Per their rationale, it should be
  // large enough to not be easily hit, but not so large that this check never fires.
  static_assert(sizeof(T)  // NOLINT(bugprone-sizeof-expression)
                  <= 256,  // NOLINT(readability-magic-numbers)
                "Trying to use the default storage size for SmallVector<T>, but using a type that "
                "is really big. If this is intentional, please explicitly set the inline storage "
                "size with SmallVector<T, N>.");

  // The following calculation is identical to what LLVM does. 64 bytes is presumably what they
  // believe to be the sweet spot between too small to be useful, and too big to efficiently
  // copy on the stack.
  constexpr std::uint32_t preferred_sizeof = 64;
  // Variant is laid out roughly like so:
  //
  // variant {
  //   union {
  //     T a;
  //     U b;
  //   } storage;
  //
  //   std::size_t idx;
  // };
  //
  // std::inplace_vector on the other hand is laid out like so:
  //
  // inplace_vector {
  //   std::size_t size;
  //   alignof(T) char data[SMALL_SIZE];
  // };
  //
  // while std::vector is laid out like so:
  //
  // vector {
  //   T *data;
  //   std::size_t cap;
  //   std::size_t size;
  // };
  //
  // (There is some variance in vector implementations between holding a pointer and 2 sizes vs
  // 2 pointers (begin, end_cap) and a size, but since sizeof(std::size_t) == sizeof(T*), both
  // implementations are equivalent in terms of storage, always 24 bytes).
  //
  // Now, the variant index is always there, and both vectors will have at least 1 std::size_t size
  // member. So to calculate the true upper bound of the inline storage such that it is roughly
  // 64 bytes we need to take the size of the entire variant and shave off 2 size_types. This
  // will give us the amount of space (up to 64) that inline storage could fill.
  constexpr std::uint32_t potential_inline_storage_capacity =
    // A.K.A sizeof(storage_type), the size of the variant
    sizeof(SmallVector<T, 0>)
    // The size of the variant index
    - sizeof(decltype(std::declval<typename SmallVector<T, 0>::storage_type&>().index()))
    // The size of the common std::size_t size member for both vectors
    - sizeof(typename SmallVector<T, 0>::size_type);

  static_assert(potential_inline_storage_capacity <= preferred_sizeof);

  constexpr std::uint32_t preferred_inline_bytes =
    preferred_sizeof - potential_inline_storage_capacity;
  constexpr std::uint32_t num_elements_that_fit =
    preferred_inline_bytes / sizeof(T);  // NOLINT(bugprone-sizeof-expression)

  return num_elements_that_fit == 0 ? 1 : num_elements_that_fit;
}

}  // namespace small_vector_detail

namespace tags {

struct iterator_tag_t {};

inline constexpr iterator_tag_t iterator_tag{};  // NOLINT(readability-identifier-naming)

struct size_tag_t {};

inline constexpr size_tag_t size_tag{};  // NOLINT(readability-identifier-naming)

}  // namespace tags

/**
 * @brief A vector-like sequence container with small-size optimization, similar to LLVM's
 * SmallVector.
 *
 * @tparam T Element type.
 * @tparam SmallSize Number of elements to store inline before falling back to heap allocation.
 * Defaults to an internal heuristic.
 *
 * `SmallVector` behaves like `std::vector` in terms of API and exception guarantees, including
 * iterator invalidation on any mutating operation. It is designed for performance-critical
 * scenarios where small sequences are common, by storing up to `SmallSize` elements inline on
 * the stack. This avoids heap allocation and improves cache locality for short vectors.
 *
 * When the number of elements exceeds `SmallSize`, it transparently falls back to heap
 * allocation, preserving the semantics of `std::vector`.
 *
 * In general, users should not specify `SmallSize` unless they have specific knowledge of
 * their workload. The default value is chosen to balance memory usage and performance.
 */
template <typename T, std::uint32_t SmallSize = small_vector_detail::storage_size<T>()>
class SmallVector {
  template <typename U>
  friend constexpr std::uint32_t small_vector_detail::storage_size();

  using small_storage_type = ::cuda::std::inplace_vector<T, SmallSize>;
  using big_storage_type   = std::vector<T>;
  // Order of entries in the variant is important. We want the variant to start
  // in "small" mode when default constructed, so small_storage_type should come
  // first.
  using storage_type = std::variant<small_storage_type, big_storage_type>;

 public:
  using value_type      = T;
  using difference_type = std::common_type_t<typename small_storage_type::difference_type,
                                             typename big_storage_type::difference_type>;
  using size_type       = std::common_type_t<typename small_storage_type::size_type,
                                             typename big_storage_type::size_type>;

  // Cannot use the common types here because std::vector uses allocator_traits
  // to derive these, while inplace_vector does not. So they might not agree.
  using reference       = value_type&;
  using const_reference = const value_type&;
  using pointer         = value_type*;
  using const_pointer   = const value_type*;
  // We use the raw pointers as iterators here because inplace_vector and
  // std::vector technically have different iterator types. Making yet another
  // iterator wrapper over them seems silly when we know that a vector's
  // iterator is just a raw pointer.
  using iterator               = pointer;
  using const_iterator         = const_pointer;
  using reverse_iterator       = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  /**
   * @brief Returns the statically defined small buffer size.
   *
   * @return The small buffer size in elements.
   */
  [[nodiscard]] static constexpr std::uint32_t small_capacity() noexcept;

  SmallVector()                              = default;
  SmallVector(const SmallVector&)            = default;
  SmallVector& operator=(const SmallVector&) = default;
  // If the variant throws in move ctor (which it cannot, but compilers won't figure that out),
  // we do not care, we would rather abort the program. If we remove noexcept here, then every
  // STL object will opt to copy SmallVector rather than move it whenever they need to, which
  // is not ideal.
  // NOLINTBEGIN(bugprone-exception-escape)
  SmallVector(SmallVector&&) noexcept            = default;
  SmallVector& operator=(SmallVector&&) noexcept = default;
  // NOLINTEND(bugprone-exception-escape)

  /**
   * @brief Constructs a `SmallVector` with a given size, filled with the specified value.
   *
   * @param tag Tag to disambiguate constructor overloads.
   * @param count Number of elements to initialize.
   * @param value Value to fill the vector with.
   */
  SmallVector(tags::size_tag_t, size_type count, const value_type& value);

  /**
   * @brief Constructs a `SmallVector` with elements in the given iterator range.
   *
   * @param tag Tag to disambiguate constructor overloads.
   * @param begin Iterator to the beginning of the range.
   * @param end Iterator to the end of the range.
   */
  template <typename It>
  SmallVector(tags::iterator_tag_t, It begin, It end);

  /**
   * @brief Constructs a `SmallVector` from an initializer list.
   *
   * @param init The initializer list to copy elements from.
   */
  SmallVector(std::initializer_list<value_type> init);

  /**
   * @brief Constructs a `SmallVector` from a span of values.
   *
   * @param span The span to copy elements from.
   */
  explicit SmallVector(Span<const value_type> span);

  /**
   * @brief Constructs a `SmallVector` from a `std::vector`.
   *
   * @param vec The vector construct from.
   */
  explicit SmallVector(std::vector<value_type> vec);

  /**
   * @brief Replaces the contents with `count` copies of the specified `value`.
   *
   * @param count Number of elements to assign.
   * @param value Value to assign to each element.
   */
  void assign(tags::size_tag_t, size_type count, const value_type& value);

  template <typename It>
  void assign(tags::iterator_tag_t, It begin, It end);

  /**
   * @brief Returns a reference to the element at the specified `pos`, with bounds checking.
   *
   * @param pos Index of the element to access.
   *
   * @return Reference to the element at `pos`.
   *
   * @throw std::out_of_range If `pos` is not in range of the vector.
   */
  [[nodiscard]] reference at(size_type pos);

  /**
   * @brief Returns a const reference to the element at the specified `pos`, with bounds checking.
   *
   * @param pos Index of the element to access.
   *
   * @return Const reference to the element at `pos`.
   *
   * @throw std::out_of_range If `pos` is not in range of the vector.
   */
  [[nodiscard]] const_reference at(size_type pos) const;

  /**
   * @brief Returns a reference to the element at the specified `pos` without bounds checking.
   *
   * @param pos Index of the element to access.
   *
   * @return Reference to the element at `pos`.
   */
  [[nodiscard]] reference operator[](size_type pos) noexcept;

  /**
   * @brief Returns a const reference to the element at the specified `pos` without bounds checking.
   *
   * @param pos Index of the element to access.
   *
   * @return Const reference to the element at `pos`.
   */
  [[nodiscard]] const_reference operator[](size_type pos) const noexcept;

  /**
   * @return Reference to the first element.
   */
  [[nodiscard]] reference front() noexcept;

  /**
   * @return Const reference to the first element.
   */
  [[nodiscard]] const_reference front() const noexcept;

  /**
   * @return Reference to the last element.
   */
  [[nodiscard]] reference back() noexcept;

  /**
   * @return Const reference to the last element.
   */
  [[nodiscard]] const_reference back() const noexcept;

  /**
   * @return Pointer to the data.
   */
  [[nodiscard]] pointer data() noexcept;

  /**
   * @return Const pointer to the data.
   */
  [[nodiscard]] const_pointer data() const noexcept;

  /**
   * @return `true` if the vector contains no elements, otherwise `false`.
   */
  [[nodiscard]] bool empty() const noexcept;

  /**
   * @return The size of the vector.
   */
  [[nodiscard]] size_type size() const noexcept;

  /**
   * @brief Returns the number of elements that can be held in currently allocated storage.
   *
   * @return The capacity of the vector.
   */
  [[nodiscard]] size_type capacity() const noexcept;

  /**
   * @brief Reserves storage to accommodate at least `new_cap` elements.
   *
   * @param new_cap New capacity to reserve.
   *
   * If `new_cap` is larger than the current capacity, this routine will not just increase the
   * capacity, but will also switch the backing storage entirely if `new_cap > small_capacity()`.
   * Therefore the user should take care to reacquire any iterators or pointers to elements after
   * this routine returns.
   */
  void reserve(size_type new_cap);

  /**
   * @brief Clears the contents, leaving the capacity unchanged.
   *
   * This also leaves the backing storage unchanged, i.e. calling this routine will not switch
   * from dynamic to static storage.
   */
  void clear();

  /**
   * @brief Inserts a copy of `value` before `pos`.
   *
   * @param pos Position to insert before.
   * @param value Value to copy-insert.
   *
   * @return Iterator to the inserted element.
   */
  iterator insert(const_iterator pos, const value_type& value);

  /**
   * @brief Inserts a moved `value` before `pos`.
   *
   * @param pos Position to insert before.
   * @param value Value to move-insert.
   *
   * @return Iterator to the inserted element.
   */
  iterator insert(const_iterator pos, value_type&& value);

  /**
   * @brief Erases the element at `pos`.
   *
   * @param pos Position of the element to erase.
   *
   * @return Iterator following the last removed element.
   */
  iterator erase(const_iterator pos);

  /**
   * @brief Erases elements in the range [`first`, `last`).
   *
   * @param first Iterator to the first element to erase.
   * @param last Iterator past the last element to erase.
   *
   * @return Iterator following the last removed element.
   */
  iterator erase(const_iterator first, const_iterator last);

  /**
   * @brief Appends a copy of `value` to the end.
   *
   * @param value Value to append.
   */
  void push_back(const value_type& value);

  /**
   * @brief Appends a moved `value` to the end.
   *
   * @param value Value to append.
   */
  void push_back(value_type&& value);

  /**
   * @brief Constructs an element in-place at the end.
   *
   * @param args Arguments forwarded to the element constructor.
   *
   * @return Reference to the newly constructed element.
   */
  template <typename... Args>
  reference emplace_back(Args&&... args);

  /**
   * @brief Removes the last element.
   */
  void pop_back();

  /**
   * @return Iterator to the first element.
   */
  [[nodiscard]] iterator begin() noexcept;

  /**
   * @return Const iterator to the first element.
   */
  [[nodiscard]] const_iterator begin() const noexcept;

  /**
   * @return Const iterator to the first element.
   */
  [[nodiscard]] const_iterator cbegin() const noexcept;

  /**
   * @return Iterator to one past the last element.
   */
  [[nodiscard]] iterator end() noexcept;

  /**
   * @return Const iterator to one past the last element.
   */
  [[nodiscard]] const_iterator end() const noexcept;

  /**
   * @return Const iterator to one past the last element.
   */
  [[nodiscard]] const_iterator cend() const noexcept;

  /**
   * @return Reverse iterator to the last element.
   */
  [[nodiscard]] reverse_iterator rbegin() noexcept;

  /**
   * @return Const teverse iterator to the last element.
   */
  [[nodiscard]] const_reverse_iterator rbegin() const noexcept;

  /**
   * @return Const teverse iterator to the last element.
   */
  [[nodiscard]] const_reverse_iterator crbegin() const noexcept;

  /**
   * @return Reverse iterator to one before the first element.
   */
  [[nodiscard]] reverse_iterator rend() noexcept;

  /**
   * @return Const reverse iterator to one before the first element.
   */
  [[nodiscard]] const_reverse_iterator rend() const noexcept;

  /**
   * @return Const reverse iterator to one before the first element.
   */
  [[nodiscard]] const_reverse_iterator crend() const noexcept;

  /**
   * @return Hash value of the vectors contents.
   */
  [[nodiscard]] std::size_t hash() const noexcept;

 private:
  /**
   * @brief Converts a `SmallVector::const_iterator` to an iterator of the underlying storage
   * type.
   *
   * @param storage_begin Iterator to the beginning of one of the storages.
   * @param pos Iterator to convert.
   *
   * @return Converted iterator.
   */
  template <typename It>
  [[nodiscard]] static It convert_iterator_(It storage_begin, const_iterator pos);

  /**
   * @return Const reference to storage.
   */
  [[nodiscard]] const storage_type& storage_() const noexcept;

  /**
   * @return Reference to storage.
   */
  [[nodiscard]] storage_type& storage_() noexcept;

  /**
   * @brief Converts internal storage to big storage with capacity `target_cap`.
   *
   * @param target_cap Desired capacity.
   *
   * @return Reference to the new big storage.
   */
  big_storage_type& convert_to_big_storage_(size_type target_cap);

  template <typename ValueType>
  void push_back_impl_(ValueType&& value);

  template <typename ValueType>
  [[nodiscard]] iterator insert_impl_(const_iterator pos, ValueType&& value);

  storage_type data_{};
};

// ==========================================================================================

template <typename T>
SmallVector(tags::size_tag_t, std::size_t, const T&) -> SmallVector<T>;

template <typename It>
SmallVector(tags::iterator_tag_t, It, It)
  -> SmallVector<typename std::iterator_traits<It>::value_type>;

template <typename T>
SmallVector(Span<const T>) -> SmallVector<T>;

template <typename T>
SmallVector(std::vector<T>) -> SmallVector<T>;

template <typename T>
SmallVector(std::initializer_list<T>) -> SmallVector<T>;

// ==========================================================================================

template <typename T, std::uint32_t S, std::uint32_t S2>
[[nodiscard]] bool operator==(const SmallVector<T, S>& x, const SmallVector<T, S2>& y);

template <typename T, std::uint32_t S, std::uint32_t S2>
[[nodiscard]] bool operator!=(const SmallVector<T, S>& x, const SmallVector<T, S2>& y);

template <typename T, std::uint32_t S, std::uint32_t S2>
[[nodiscard]] bool operator<(const SmallVector<T, S>& x, const SmallVector<T, S2>& y);

template <typename T, std::uint32_t S, std::uint32_t S2>
[[nodiscard]] bool operator>(const SmallVector<T, S>& x, const SmallVector<T, S2>& y);

template <typename T, std::uint32_t S, std::uint32_t S2>
[[nodiscard]] bool operator>=(const SmallVector<T, S>& x, const SmallVector<T, S2>& y);

template <typename T, std::uint32_t S, std::uint32_t S2>
[[nodiscard]] bool operator<=(const SmallVector<T, S>& x, const SmallVector<T, S2>& y);

}  // namespace legate::detail

#include <legate/utilities/detail/small_vector.inl>
