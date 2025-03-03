/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <iosfwd>
#include <memory>
#include <string>
#include <string_view>

namespace legate::detail {

template <typename CharT, typename TraitsT = std::char_traits<CharT>>
class BasicZStringView : private std::basic_string_view<CharT, TraitsT> {
 public:
  using base_view_type = std::basic_string_view<CharT, TraitsT>;

  using typename base_view_type::const_pointer;
  using typename base_view_type::const_reference;
  using typename base_view_type::pointer;
  using typename base_view_type::reference;
  using typename base_view_type::traits_type;
  using typename base_view_type::value_type;

  using typename base_view_type::const_iterator;
  using typename base_view_type::const_reverse_iterator;
  using typename base_view_type::iterator;
  using typename base_view_type::reverse_iterator;

  using typename base_view_type::difference_type;
  using typename base_view_type::size_type;

  using base_view_type::npos;

  constexpr BasicZStringView() noexcept;

  // NOLINTBEGIN(google-explicit-constructor)
  constexpr BasicZStringView(const CharT* str) noexcept;

  template <std::size_t N>
  constexpr BasicZStringView(const CharT (&str)[N]) noexcept;

  constexpr BasicZStringView(const std::basic_string<CharT, TraitsT>& str) noexcept;
  // NOLINTEND(google-explicit-constructor)

  // Disallow this because str might not be null terminated
  BasicZStringView(const CharT* str, size_type len) = delete;
  // Disallow because this is obviously not null terminated
  BasicZStringView(std::nullptr_t) = delete;
  // Disallow because the view may not be null terminated
  BasicZStringView(const base_view_type& view) = delete;

  constexpr BasicZStringView(const BasicZStringView&) noexcept            = default;
  constexpr BasicZStringView& operator=(const BasicZStringView&) noexcept = default;
  constexpr BasicZStringView(BasicZStringView&&) noexcept                 = default;
  constexpr BasicZStringView& operator=(BasicZStringView&&) noexcept      = default;

  using base_view_type::begin;
  using base_view_type::cbegin;
  using base_view_type::cend;
  using base_view_type::crbegin;
  using base_view_type::crend;
  using base_view_type::end;
  using base_view_type::rbegin;
  using base_view_type::rend;

  using base_view_type::empty;
  using base_view_type::length;
  using base_view_type::max_size;
  using base_view_type::size;

  using base_view_type::operator[];
  using base_view_type::at;
  using base_view_type::back;
  using base_view_type::data;
  using base_view_type::front;

  using base_view_type::remove_prefix;

  template <typename Allocator = std::allocator<CharT>>
  [[nodiscard]] std::basic_string<CharT, TraitsT, Allocator> to_string(
    const Allocator& a = Allocator{}) const;

  [[nodiscard]] constexpr base_view_type as_string_view() const noexcept;

  using base_view_type::compare;
  using base_view_type::copy;
  using base_view_type::substr;
};

template <typename C, typename T>
std::basic_ostream<C, T>& operator<<(std::basic_ostream<C, T>& os, BasicZStringView<C, T> sv);

template <typename C, typename T>
constexpr bool operator==(BasicZStringView<C, T> lhs, BasicZStringView<C, T> rhs);

template <typename C, typename T>
constexpr bool operator!=(BasicZStringView<C, T> lhs, BasicZStringView<C, T> rhs);

template <typename C, typename T>
constexpr bool operator==(typename BasicZStringView<C, T>::base_view_type lhs,
                          BasicZStringView<C, T> rhs);

template <typename C, typename T>
constexpr bool operator!=(typename BasicZStringView<C, T>::base_view_type lhs,
                          BasicZStringView<C, T> rhs);

template <typename C, typename T>
constexpr bool operator==(BasicZStringView<C, T> lhs,
                          typename BasicZStringView<C, T>::base_view_type rhs);

template <typename C, typename T>
constexpr bool operator!=(BasicZStringView<C, T> lhs,
                          typename BasicZStringView<C, T>::base_view_type rhs);

}  // namespace legate::detail

namespace std {

template <typename CharT, typename TraitsT>
struct hash<legate::detail::BasicZStringView<CharT, TraitsT>> {  // NOLINT(cert-dcl58-cpp)
  [[nodiscard]] constexpr std::size_t operator()(
    const legate::detail::BasicZStringView<CharT, TraitsT>& sv) const noexcept;
};

}  // namespace std

#include <legate/utilities/detail/zstring_view.inl>

namespace legate::detail {

using ZStringView = BasicZStringView<char>;

}  // namespace legate::detail
