/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/cpp_version.h>
#include <legate/utilities/detail/array_algorithms.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/detail/type_traits.h>
#include <legate/utilities/hash.h>

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace legate::detail {

LEGATE_CPP_VERSION_TODO(20, "Use std::to_address() instead");

template <typename T>
constexpr T* to_address(T* p) noexcept
{
  static_assert(!std::is_function_v<T>);
  return p;
}

template <typename T, typename = std::void_t<decltype(std::declval<T>().operator->())>>
constexpr auto* to_address(const T& p) noexcept
{
  return to_address(p.operator->());
}

// ==========================================================================================

template <typename T, std::uint32_t S>
const typename SmallVector<T, S>::storage_type& SmallVector<T, S>::storage_() const noexcept
{
  return data_;
}

template <typename T, std::uint32_t S>
typename SmallVector<T, S>::storage_type& SmallVector<T, S>::storage_() noexcept
{
  return data_;
}

template <typename T, std::uint32_t S>
typename SmallVector<T, S>::big_storage_type& SmallVector<T, S>::convert_to_big_storage_(
  size_type target_cap)
{
  // Use a temporary vector here for 2 reasons:
  //
  // 1. We cannot clobber the small storage before we copy out of it.
  // 2. We need to be exception safe, so shouldn't leave our storage half constructed in case
  //    things go wrong.
  auto vec    = big_storage_type{};
  auto& small = std::get<small_storage_type>(storage_());

  vec.reserve(target_cap);
  if constexpr (std::is_nothrow_move_constructible_v<T>) {
    vec.assign(std::move_iterator{small.begin()}, std::move_iterator{small.end()});
  } else {
    vec.assign(small.begin(), small.end());
  }
  return storage_().template emplace<big_storage_type>(std::move(vec));
}

template <typename T, std::uint32_t S>
template <typename It>
/* static */ It SmallVector<T, S>::convert_iterator_(It storage_begin, const_iterator pos)
{
  // Our iterators are not going to be the same types as our storages, so we need to translate
  // them. Luckily, since both storages are just linear containers (and their iterators barely
  // more than simple pointer wrappers), it suffices to do some pointer arithmetic to do the
  // conversion.
  //
  // Need an explicit cast to const ref in case the storage iterator is not const.
  const_reference ref = *storage_begin;

  return storage_begin + std::distance(std::addressof(ref), to_address(pos));
}

// ------------------------------------------------------------------------------------------

template <typename T, std::uint32_t S>
/* static */ constexpr std::uint32_t SmallVector<T, S>::small_capacity() noexcept
{
  // Should be small_storage_type::capacity(), but work around CCCL bug
  // https://github.com/NVIDIA/cccl/issues/5142
  return S;
}

// ------------------------------------------------------------------------------------------

template <typename T, std::uint32_t S>
SmallVector<T, S>::SmallVector(tags::size_tag_t, size_type count, const value_type& value)
{
  if (count <= small_capacity()) {
    storage_().template emplace<small_storage_type>(count, value);
  } else {
    storage_().template emplace<big_storage_type>(count, value);
  }
}

template <typename T, std::uint32_t S>
template <typename It>
SmallVector<T, S>::SmallVector(tags::iterator_tag_t, It begin, It end)
{
  if constexpr (std::is_base_of_v<std::random_access_iterator_tag,
                                  typename std::iterator_traits<It>::iterator_category>) {
    const auto count = static_cast<size_type>(std::distance(begin, end));

    if (count <= small_capacity()) {
      storage_().template emplace<small_storage_type>(begin, end);
    } else {
      storage_().template emplace<big_storage_type>(begin, end);
    }
  } else {
    std::copy(begin, end, std::back_inserter(*this));
  }
}

template <typename T, std::uint32_t S>
SmallVector<T, S>::SmallVector(std::initializer_list<value_type> init)
  : SmallVector{tags::iterator_tag, std::begin(init), std::end(init)}
{
}

// ------------------------------------------------------------------------------------------

template <typename T, std::uint32_t S>
SmallVector<T, S>::SmallVector(Span<const value_type> span)
  : SmallVector{tags::iterator_tag, span.begin(), span.end()}
{
}

// ------------------------------------------------------------------------------------------

template <typename T, std::uint32_t S>
SmallVector<T, S>::SmallVector(std::vector<value_type> vec)
  : data_{std::in_place_type_t<big_storage_type>{}, std::move(vec)}
{
}

// ------------------------------------------------------------------------------------------

template <typename T, std::uint32_t S>
void SmallVector<T, S>::assign(tags::size_tag_t, size_type count, const value_type& value)
{
  // The reserve call will do the right thing to switch us between storages.
  clear();
  reserve(count);
  std::visit([&](auto&& st) { st.assign(count, value); }, storage_());
}

template <typename T, std::uint32_t S>
template <typename It>
void SmallVector<T, S>::assign(tags::iterator_tag_t, It begin, It end)
{
  clear();
  if constexpr (std::is_base_of_v<std::random_access_iterator_tag,
                                  typename std::iterator_traits<It>::iterator_category>) {
    const auto count = std::distance(begin, end);

    reserve(count);
    std::visit([&](auto&& st) { st.assign(begin, end); }, storage_());
  } else {
    std::copy(begin, end, std::back_inserter(*this));
  }
}

// ------------------------------------------------------------------------------------------

template <typename T, std::uint32_t S>
typename SmallVector<T, S>::reference SmallVector<T, S>::at(size_type pos)
{
  return std::visit(
    Overload{[&](small_storage_type& st) -> reference {
               // Work around CCCL bug https://github.com/NVIDIA/cccl/issues/5294
               if (pos == st.size()) {
                 throw std::out_of_range{"inplace_vector::at"};  // legate-lint: no-traced-throw
               }
               return st.at(pos);
             },
             [&](big_storage_type& st) -> reference { return st.at(pos); }},
    storage_());
}

template <typename T, std::uint32_t S>
typename SmallVector<T, S>::const_reference SmallVector<T, S>::at(size_type pos) const
{
  return std::visit(
    Overload{[&](const small_storage_type& st) -> const_reference {
               // Work around CCCL bug https://github.com/NVIDIA/cccl/issues/5294
               if (pos == st.size()) {
                 throw std::out_of_range{"inplace_vector::at"};  // legate-lint: no-traced-throw
               }
               return st.at(pos);
             },
             [&](const big_storage_type& st) -> const_reference { return st.at(pos); }},
    storage_());
}

// ------------------------------------------------------------------------------------------

template <typename T, std::uint32_t S>
typename SmallVector<T, S>::reference
SmallVector<T, S>::operator[](  // NOLINT(bugprone-exception-escape)
  size_type pos) noexcept
{
  return std::visit([&](auto&& st) -> reference { return st[pos]; }, storage_());
}

template <typename T, std::uint32_t S>
typename SmallVector<T, S>::const_reference
SmallVector<T, S>::operator[](  // NOLINT(bugprone-exception-escape)
  size_type pos) const noexcept
{
  return std::visit([&](auto&& st) -> const_reference { return st[pos]; }, storage_());
}

// ------------------------------------------------------------------------------------------

template <typename T, std::uint32_t S>
typename SmallVector<T, S>::reference
SmallVector<T, S>::front()  // NOLINT(bugprone-exception-escape)
  noexcept
{
  return std::visit([](auto&& st) -> reference { return st.front(); }, storage_());
}

template <typename T, std::uint32_t S>
typename SmallVector<T, S>::const_reference
SmallVector<T, S>::front()  // NOLINT(bugprone-exception-escape)
  const noexcept
{
  return std::visit([](auto&& st) -> const_reference { return st.front(); }, storage_());
}

// ------------------------------------------------------------------------------------------

template <typename T, std::uint32_t S>
typename SmallVector<T, S>::reference
SmallVector<T, S>::back()  // NOLINT(bugprone-exception-escape)
  noexcept
{
  return std::visit([](auto&& st) -> reference { return st.back(); }, storage_());
}

template <typename T, std::uint32_t S>
typename SmallVector<T, S>::const_reference
SmallVector<T, S>::back()  // NOLINT(bugprone-exception-escape)
  const noexcept
{
  return std::visit([](auto&& st) -> const_reference { return st.back(); }, storage_());
}

// ------------------------------------------------------------------------------------------

template <typename T, std::uint32_t S>
typename SmallVector<T, S>::pointer SmallVector<T, S>::data()  // NOLINT(bugprone-exception-escape)
  noexcept
{
  return std::visit([](auto&& st) { return st.data(); }, storage_());
}

template <typename T, std::uint32_t S>
typename SmallVector<T, S>::const_pointer
SmallVector<T, S>::data()  // NOLINT(bugprone-exception-escape)
  const noexcept
{
  return std::visit([](auto&& st) { return st.data(); }, storage_());
}

// ------------------------------------------------------------------------------------------

template <typename T, std::uint32_t S>
bool SmallVector<T, S>::empty()  // NOLINT(bugprone-exception-escape)
  const noexcept
{
  return std::visit([](auto&& st) { return st.empty(); }, storage_());
}

template <typename T, std::uint32_t S>
typename SmallVector<T, S>::size_type
SmallVector<T, S>::size()  // NOLINT(bugprone-exception-escape)
  const noexcept
{
  return std::visit([](auto&& st) { return st.size(); }, storage_());
}

template <typename T, std::uint32_t S>
typename SmallVector<T, S>::size_type
SmallVector<T, S>::capacity()  // NOLINT(bugprone-exception-escape)
  const noexcept
{
  return std::visit([](auto&& st) { return st.capacity(); }, storage_());
}

// ------------------------------------------------------------------------------------------

template <typename T, std::uint32_t S>
void SmallVector<T, S>::reserve(size_type new_cap)
{
  std::visit(Overload{[&](small_storage_type& small) {
                        if (new_cap > small.capacity()) {
                          convert_to_big_storage_(new_cap);
                        }
                      },
                      [&](big_storage_type& big) { big.reserve(new_cap); }},
             storage_());
}

// ------------------------------------------------------------------------------------------

template <typename T, std::uint32_t S>
void SmallVector<T, S>::clear()
{
  return std::visit([](auto&& st) { st.clear(); }, storage_());
}

// ------------------------------------------------------------------------------------------

template <typename T, std::uint32_t S>
template <typename ValueType>
typename SmallVector<T, S>::iterator SmallVector<T, S>::insert_impl_(const_iterator pos,
                                                                     ValueType&& value)
{
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    // size + 1 here because we are allowed to insert at the end
    assert_in_range(size() + 1, std::distance(cbegin(), pos));
  }

  return std::visit(
    Overload{[&](small_storage_type& small) {
               const auto small_size = small.size();

               if (small_size == small.capacity()) {
                 const auto dist = std::distance(to_address(small.cbegin()), to_address(pos));

                 // We set the target capacity to double here because we want the new vector
                 // to "remember" the growth-rate of the small vector (which for all major
                 // implementations is always 2N).
                 auto& big = convert_to_big_storage_(small_size * 2);
                 // Don't use convert_iterator_() here, because pos is still pointing into small.
                 const auto new_it =
                   big.insert(big.cbegin() + dist, std::forward<ValueType>(value));

                 return to_address(new_it);
               }
               const auto new_it = small.insert(convert_iterator_(small.cbegin(), pos),
                                                std::forward<ValueType>(value));

               return to_address(new_it);
             },
             [&](big_storage_type& big) {
               const auto new_it =
                 big.insert(convert_iterator_(big.cbegin(), pos), std::forward<ValueType>(value));

               return to_address(new_it);
             }},
    storage_());
}

template <typename T, std::uint32_t S>
typename SmallVector<T, S>::iterator SmallVector<T, S>::insert(const_iterator pos,
                                                               const value_type& value)
{
  return insert_impl_(pos, value);
}

template <typename T, std::uint32_t S>
typename SmallVector<T, S>::iterator SmallVector<T, S>::insert(const_iterator pos,
                                                               value_type&& value)
{
  return insert_impl_(pos, std::move(value));
}

// ------------------------------------------------------------------------------------------

template <typename T, std::uint32_t S>
typename SmallVector<T, S>::iterator SmallVector<T, S>::erase(const_iterator pos)
{
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    assert_in_range(size(), std::distance(cbegin(), pos));
  }

  return std::visit(
    [&](auto&& st) {
      const auto new_it = st.erase(convert_iterator_(st.cbegin(), pos));

      return to_address(new_it);
    },
    storage_());
}

template <typename T, std::uint32_t S>
typename SmallVector<T, S>::iterator SmallVector<T, S>::erase(const_iterator first,
                                                              const_iterator last)
{
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    // Size + 1 because we can be erasing with (and up to) end().
    assert_in_range(size() + 1, std::distance(cbegin(), first));
    assert_in_range(size() + 1, std::distance(cbegin(), last));
  }

  const auto dist = std::distance(first, last);

  return std::visit(
    [&](auto&& st) {
      const auto st_begin = convert_iterator_(st.begin(), first);
      const auto new_it   = st.erase(st_begin, st_begin + dist);

      return to_address(new_it);
    },
    storage_());
}

// ------------------------------------------------------------------------------------------

template <typename T, std::uint32_t S>
template <typename ValueType>
void SmallVector<T, S>::push_back_impl_(ValueType&& value)
{
  std::visit(
    Overload{[&](small_storage_type& small) {
               if (const auto small_size = small.size(); small_size == small.capacity()) {
                 // We set the target capacity to double here because we want the new vector
                 // to "remember" the growth-rate of the small vector (which for all major
                 // implementations is always 2N).
                 convert_to_big_storage_(small_size * 2).push_back(std::forward<ValueType>(value));
               } else {
                 small.unchecked_push_back(std::forward<ValueType>(value));
               }
             },
             [&](big_storage_type& big) { big.push_back(std::forward<ValueType>(value)); }},
    storage_());
}

template <typename T, std::uint32_t S>
void SmallVector<T, S>::push_back(const value_type& value)
{
  push_back_impl_(value);
}

template <typename T, std::uint32_t S>
void SmallVector<T, S>::push_back(value_type&& value)
{
  push_back_impl_(std::move(value));
}

// ------------------------------------------------------------------------------------------

template <typename T, std::uint32_t S>
template <typename... Args>
typename SmallVector<T, S>::reference SmallVector<T, S>::emplace_back(Args&&... args)
{
  return std::visit(
    Overload{
      [&](small_storage_type& small) -> reference {
        if (const auto small_size = small.size(); small_size == small.capacity()) {
          // We set the target capacity to double here because we want the new vector to
          // "remember" the growth-rate of the small vector (which for all major implementations is
          // always 2N).
          return convert_to_big_storage_(small_size * 2).emplace_back(std::forward<Args>(args)...);
        }
        return small.unchecked_emplace_back(std::forward<Args>(args)...);
      },
      [&](big_storage_type& big) -> reference {
        return big.emplace_back(std::forward<Args>(args)...);
      }},
    storage_());
}

// ------------------------------------------------------------------------------------------

template <typename T, std::uint32_t S>
void SmallVector<T, S>::pop_back()
{
  std::visit([](auto&& st) { st.pop_back(); }, storage_());
}

// ------------------------------------------------------------------------------------------

template <typename T, std::uint32_t S>
typename SmallVector<T, S>::iterator
SmallVector<T, S>::begin()  // NOLINT(bugprone-exception-escape)
  noexcept
{
  return std::visit([](auto&& st) -> iterator { return to_address(st.begin()); }, storage_());
}

template <typename T, std::uint32_t S>
typename SmallVector<T, S>::const_iterator
SmallVector<T, S>::begin()  // NOLINT(bugprone-exception-escape)
  const noexcept
{
  return cbegin();
}

template <typename T, std::uint32_t S>
typename SmallVector<T, S>::const_iterator
SmallVector<T, S>::cbegin()  // NOLINT(bugprone-exception-escape)
  const noexcept
{
  return std::visit([](auto&& st) -> const_iterator { return to_address(st.cbegin()); },
                    storage_());
}

// ------------------------------------------------------------------------------------------

template <typename T, std::uint32_t S>
typename SmallVector<T, S>::iterator SmallVector<T, S>::end()  // NOLINT(bugprone-exception-escape)
  noexcept
{
  return std::visit([](auto&& st) -> iterator { return to_address(st.end()); }, storage_());
}

template <typename T, std::uint32_t S>
typename SmallVector<T, S>::const_iterator
SmallVector<T, S>::end()  // NOLINT(bugprone-exception-escape)
  const noexcept
{
  return cend();
}

template <typename T, std::uint32_t S>
typename SmallVector<T, S>::const_iterator
SmallVector<T, S>::cend()  // NOLINT(bugprone-exception-escape)
  const noexcept
{
  return std::visit([](auto&& st) -> const_iterator { return to_address(st.cend()); }, storage_());
}

// ------------------------------------------------------------------------------------------

template <typename T, std::uint32_t S>
typename SmallVector<T, S>::reverse_iterator
SmallVector<T, S>::rbegin()  // NOLINT(bugprone-exception-escape)
  noexcept
{
  return std::visit([](auto&& st) -> reverse_iterator { return to_address(st.rbegin()); },
                    storage_());
}

template <typename T, std::uint32_t S>
typename SmallVector<T, S>::const_reverse_iterator
SmallVector<T, S>::rbegin()  // NOLINT(bugprone-exception-escape)
  const noexcept
{
  return crbegin();
}

template <typename T, std::uint32_t S>
typename SmallVector<T, S>::const_reverse_iterator
SmallVector<T, S>::crbegin()  // NOLINT(bugprone-exception-escape)
  const noexcept
{
  return std::visit([](auto&& st) -> const_reverse_iterator { return to_address(st.crbegin()); },
                    storage_());
}

// ------------------------------------------------------------------------------------------

template <typename T, std::uint32_t S>
typename SmallVector<T, S>::reverse_iterator
SmallVector<T, S>::rend()  // NOLINT(bugprone-exception-escape)
  noexcept
{
  return std::visit([](auto&& st) -> reverse_iterator { return to_address(st.rend()); },
                    storage_());
}

template <typename T, std::uint32_t S>
typename SmallVector<T, S>::const_reverse_iterator
SmallVector<T, S>::rend()  // NOLINT(bugprone-exception-escape)
  const noexcept
{
  return crend();
}

template <typename T, std::uint32_t S>
typename SmallVector<T, S>::const_reverse_iterator
SmallVector<T, S>::crend()  // NOLINT(bugprone-exception-escape)
  const noexcept
{
  return std::visit([](auto&& st) -> const_reverse_iterator { return to_address(st.crend()); },
                    storage_());
}

// ------------------------------------------------------------------------------------------

template <typename T, std::uint32_t S>
std::size_t SmallVector<T, S>::hash() const noexcept
{
  std::size_t result = 0;

  for (auto&& v : *this) {
    hash_combine(result, v);
  }
  return result;
}

// ==========================================================================================

template <typename T, std::uint32_t S, std::uint32_t S2>
bool operator==(const SmallVector<T, S>& x, const SmallVector<T, S2>& y)
{
  return (std::addressof(x) == std::addressof(y)) ||
         std::equal(x.begin(), x.end(), y.begin(), y.end());
}

template <typename T, std::uint32_t S, std::uint32_t S2>
bool operator!=(const SmallVector<T, S>& x, const SmallVector<T, S2>& y)
{
  return !(x == y);
}

template <typename T, std::uint32_t S, std::uint32_t S2>
bool operator<(const SmallVector<T, S>& x, const SmallVector<T, S2>& y)
{
  return std::lexicographical_compare(x.begin(), x.end(), y.begin(), y.end());
}

template <typename T, std::uint32_t S, std::uint32_t S2>
bool operator>(const SmallVector<T, S>& x, const SmallVector<T, S2>& y)
{
  return y < x;
}

template <typename T, std::uint32_t S, std::uint32_t S2>
bool operator>=(const SmallVector<T, S>& x, const SmallVector<T, S2>& y)
{
  return !(x < y);
}

template <typename T, std::uint32_t S, std::uint32_t S2>
bool operator<=(const SmallVector<T, S>& x, const SmallVector<T, S2>& y)
{
  return !(y < x);
}

}  // namespace legate::detail
