/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/zstring_view.h>

#include <type_traits>

namespace legate::detail {

template <typename C, typename T>
constexpr BasicZStringView<C, T>::BasicZStringView() noexcept : base_view_type{""}
{
}

// Use "CharT" here because otherwise doxygen errors:
//
// error: no matching class member found for
// template < C, T >
// constexpr legate::detail::BasicZStringView< C, T >::BasicZStringView(const C *str)
//
// Because it just tries to match the literal source I guess?
template <typename CharT, typename T>
constexpr BasicZStringView<CharT, T>::BasicZStringView(const CharT* str) noexcept
  : base_view_type{str}
{
}

template <typename C, typename T>
template <std::size_t N>
constexpr BasicZStringView<C, T>::BasicZStringView(const C (&str)[N]) noexcept
  : base_view_type{static_cast<const C*>(str), N}
{
}

// Same here for doxygen...
template <typename CharT, typename TraitsT>
constexpr BasicZStringView<CharT, TraitsT>::BasicZStringView(
  const std::basic_string<CharT, TraitsT>& str) noexcept
  : base_view_type{str}
{
}

template <typename C, typename T>
template <typename Allocator>
[[nodiscard]] std::basic_string<C, T, Allocator> BasicZStringView<C, T>::to_string(
  const Allocator& a) const
{
  return {begin(), end(), a};
}

template <typename C, typename T>
constexpr typename BasicZStringView<C, T>::base_view_type BasicZStringView<C, T>::as_string_view()
  const noexcept
{
  return static_cast<base_view_type>(*this);
}

// ==========================================================================================

template <typename C, typename T>
std::basic_ostream<C, T>& operator<<(std::basic_ostream<C, T>& os, BasicZStringView<C, T> sv)
{
  os << sv.as_string_view();
  return os;
}

template <typename C, typename T>
constexpr bool operator==(BasicZStringView<C, T> lhs, BasicZStringView<C, T> rhs)
{
  return lhs.as_string_view() == rhs.as_string_view();
}

template <typename C, typename T>
constexpr bool operator!=(BasicZStringView<C, T> lhs, BasicZStringView<C, T> rhs)
{
  return !(lhs == rhs);
}

template <typename C, typename T>
constexpr bool operator==(typename BasicZStringView<C, T>::base_view_type lhs,
                          BasicZStringView<C, T> rhs)
{
  return lhs == rhs.as_string_view();
}

template <typename C, typename T>
constexpr bool operator!=(typename BasicZStringView<C, T>::base_view_type lhs,
                          BasicZStringView<C, T> rhs)
{
  return !(lhs == rhs);
}

template <typename C, typename T>
constexpr bool operator==(BasicZStringView<C, T> lhs,
                          typename BasicZStringView<C, T>::base_view_type rhs)
{
  return lhs.as_string_view() == rhs;
}

template <typename C, typename T>
constexpr bool operator!=(BasicZStringView<C, T> lhs,
                          typename BasicZStringView<C, T>::base_view_type rhs)
{
  return !(lhs == rhs);
}

}  // namespace legate::detail

namespace std {

template <typename CharT, typename TraitsT>
[[nodiscard]] constexpr std::size_t
hash<legate::detail::BasicZStringView<CharT, TraitsT>>::operator()(
  const legate::detail::BasicZStringView<CharT, TraitsT>& sv) const noexcept
{
  using zsv_type = std::decay_t<decltype(sv)>;
  using sv_type  = typename zsv_type::base_view_type;

  return std::hash<sv_type>{}(sv.as_string_view());
}

}  // namespace std
