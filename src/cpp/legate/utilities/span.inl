/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/assert.h>
#include <legate/utilities/cpp_version.h>
#include <legate/utilities/span.h>

#include <iterator>
#include <type_traits>

namespace legate {

namespace detail {

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

}  // namespace detail

template <typename T>
template <typename C>
constexpr Span<T>::Span(C& container, container_tag)
  : Span{std::data(container), std::size(container), size_tag{}}
{
}

template <typename T>
constexpr Span<T>::Span(T* data, size_type size, size_tag) : data_{data}, size_{size}
{
}

// ------------------------------------------------------------------------------------------

template <typename T>
template <typename C, typename SFINAE>
constexpr Span<T>::Span(C& container) : Span{container, container_tag{}}
{
}

template <typename T>
constexpr Span<T>::Span(std::initializer_list<T> il) : Span{il, container_tag{}}
{
}

template <typename T>
template <typename It>
constexpr Span<T>::Span(It begin, It end)
  : Span{detail::to_address(begin), static_cast<size_type>(std::distance(begin, end)), size_tag{}}
{
  using category = typename std::iterator_traits<It>::iterator_category;
  static_assert(std::is_convertible_v<category, std::random_access_iterator_tag>);
}

template <typename T>
constexpr Span<T>::Span(T* data, size_type size) : Span{data, size, size_tag{}}
{
}

template <typename T>
constexpr typename Span<T>::size_type Span<T>::size() const
{
  return size_;
}

template <typename T>
constexpr typename Span<T>::reference Span<T>::operator[](size_type pos) const
{
  LEGATE_ASSERT(pos < size_);
  return data_[pos];
}

template <typename T>
constexpr typename Span<T>::const_iterator Span<T>::begin() const
{
  return data_;
}

template <typename T>
constexpr typename Span<T>::const_iterator Span<T>::end() const
{
  return data_ + size_;
}

template <typename T>
constexpr typename Span<T>::iterator Span<T>::begin()
{
  return data_;
}

template <typename T>
constexpr typename Span<T>::iterator Span<T>::end()
{
  return data_ + size_;
}

template <typename T>
constexpr typename Span<T>::reverse_const_iterator Span<T>::rbegin() const
{
  return reverse_const_iterator{end()};
}

template <typename T>
constexpr typename Span<T>::reverse_const_iterator Span<T>::rend() const
{
  return reverse_const_iterator{begin()};
}

template <typename T>
constexpr typename Span<T>::reverse_iterator Span<T>::rbegin()
{
  return reverse_iterator{end()};
}

template <typename T>
constexpr typename Span<T>::reverse_iterator Span<T>::rend()
{
  return reverse_iterator{begin()};
}

template <typename T>
constexpr typename Span<T>::reference Span<T>::front() const
{
  return *begin();
}

template <typename T>
constexpr typename Span<T>::reference Span<T>::back() const
{
  return *(end() - 1);
}

template <typename T>
constexpr Span<T> Span<T>::subspan(size_type off)
{
  LEGATE_CHECK(off <= size_);
  return {data_ + off, size_ - off, size_tag{}};
}

template <typename T>
constexpr typename Span<T>::const_pointer Span<T>::ptr() const
{
  return data();
}

template <typename T>
constexpr typename Span<T>::pointer Span<T>::data() const
{
  return data_;
}

}  // namespace legate
