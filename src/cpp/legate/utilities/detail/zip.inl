/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/utilities/detail/type_traits.h>
#include <legate/utilities/detail/zip.h>
#include <legate/utilities/macros.h>

namespace legate::detail {

namespace zip_detail {

template <typename D, typename... T>
ZiperatorBase<D, T...>::ZiperatorBase(T&&... its) : iter_tup_{std::forward<T>(its)...}
{
}

template <typename D, typename... T>
typename ZiperatorBase<D, T...>::value_type ZiperatorBase<D, T...>::operator*() const
{
  return derived_().dereference_(typename derived_type::sequence{});
}

template <typename D, typename... T>
typename ZiperatorBase<D, T...>::derived_type& ZiperatorBase<D, T...>::operator++()
{
  derived_().increment_(typename derived_type::sequence{});
  return derived_();
}

template <typename D, typename... T>
typename ZiperatorBase<D, T...>::derived_type& ZiperatorBase<D, T...>::operator--()
{
  static_assert(
    std::is_convertible_v<typename derived_type::iterator_category,
                          std::bidirectional_iterator_tag>,
    "All zipped iterators must satisfy std::bidirectional_iterator in order to go backwards!");
  derived_().decrement_(typename derived_type::sequence{});
  return derived_();
}

template <typename D, typename... T>
typename ZiperatorBase<D, T...>::derived_type ZiperatorBase<D, T...>::operator+(
  difference_type n) const
{
  static_assert(
    std::is_convertible_v<typename derived_type::iterator_category,
                          std::random_access_iterator_tag>,
    "All zipped iterators must satisfy std::random_access_iterator in order to use operator+!");
  derived_type result{derived_()};

  result += n;
  return result;
}

template <typename D, typename... T>
typename ZiperatorBase<D, T...>::derived_type ZiperatorBase<D, T...>::operator-(
  difference_type n) const
{
  static_assert(
    std::is_convertible_v<typename derived_type::iterator_category,
                          std::random_access_iterator_tag>,
    "All zipped iterators must satisfy std::random_access_iterator in order to use operator-!");
  derived_type result{derived_()};

  result -= n;
  return result;
}

template <typename D, typename... T>
typename ZiperatorBase<D, T...>::difference_type ZiperatorBase<D, T...>::operator-(
  const ZiperatorBase& other) const
{
  static_assert(
    std::is_convertible_v<typename derived_type::iterator_category,
                          std::random_access_iterator_tag>,
    "All zipped iterators must satisfy std::random_access_iterator in order to use operator-!");
  // It is unclear what should happen if any of the other iterators returned a different length
  // value, so we just use the first iterator to check this.
  return std::get<0>(derived_().iters_()) - std::get<0>(other.derived_().iters_());
}

template <typename D, typename... T>
typename ZiperatorBase<D, T...>::derived_type& ZiperatorBase<D, T...>::operator+=(difference_type n)
{
  static_assert(
    std::is_convertible_v<typename derived_type::iterator_category,
                          std::random_access_iterator_tag>,
    "All zipped iterators must satisfy std::random_access_iterator in order to use operator+=!");
  derived_().pluseq_(typename derived_type::sequence{}, n);
  return derived_();
}

template <typename D, typename... T>
typename ZiperatorBase<D, T...>::derived_type& ZiperatorBase<D, T...>::operator-=(difference_type n)
{
  static_assert(
    std::is_convertible_v<typename derived_type::iterator_category,
                          std::random_access_iterator_tag>,
    "All zipped iterators must satisfy std::random_access_iterator in order to use operator-=!");
  derived_().minuseq_(typename derived_type::sequence{}, n);
  return derived_();
}

template <typename D, typename... T>
bool ZiperatorBase<D, T...>::operator<(const ZiperatorBase& other) const
{
  static_assert(
    std::is_convertible_v<typename derived_type::iterator_category,
                          std::random_access_iterator_tag>,
    "All zipped iterators must satisfy std::random_access_iterator in order to use operator<!");
  return derived_().lessthan_(typename derived_type::sequence{}, other.derived_());
}

template <typename D, typename... T>
bool ZiperatorBase<D, T...>::operator<=(const ZiperatorBase& other) const
{
  static_assert(
    std::is_convertible_v<typename derived_type::iterator_category,
                          std::random_access_iterator_tag>,
    "All zipped iterators must satisfy std::random_access_iterator in order to use operator<=!");
  return !(other.derived_() < derived_());
}

template <typename D, typename... T>
bool ZiperatorBase<D, T...>::operator>(const ZiperatorBase& other) const
{
  static_assert(
    std::is_convertible_v<typename derived_type::iterator_category,
                          std::random_access_iterator_tag>,
    "All zipped iterators must satisfy std::random_access_iterator in order to use operator>!");
  return other.derived_() < derived_();
}

template <typename D, typename... T>
bool ZiperatorBase<D, T...>::operator>=(const ZiperatorBase& other) const
{
  static_assert(
    std::is_convertible_v<typename derived_type::iterator_category,
                          std::random_access_iterator_tag>,
    "All zipped iterators must satisfy std::random_access_iterator in order to use operator>=!");
  return !(derived_() < other.derived_());
}

template <typename D, typename... T>
bool ZiperatorBase<D, T...>::operator==(const ZiperatorBase& other) const
{
  return derived_().eq_(typename derived_type::sequence{}, other.derived_());
}

template <typename D, typename... T>
bool ZiperatorBase<D, T...>::operator!=(const ZiperatorBase& other) const
{
  return !(derived_() == other.derived_());
}

// ==========================================================================================

template <typename D, typename... T>
typename ZiperatorBase<D, T...>::derived_type& ZiperatorBase<D, T...>::derived_() noexcept
{
  return static_cast<derived_type&>(*this);
}

template <typename D, typename... T>
const typename ZiperatorBase<D, T...>::derived_type& ZiperatorBase<D, T...>::derived_()
  const noexcept
{
  return static_cast<const derived_type&>(*this);
}

template <typename D, typename... T>
typename ZiperatorBase<D, T...>::iter_tuple_type& ZiperatorBase<D, T...>::iters_() noexcept
{
  return iter_tup_;
}

template <typename D, typename... T>
const typename ZiperatorBase<D, T...>::iter_tuple_type& ZiperatorBase<D, T...>::iters_()
  const noexcept
{
  return iter_tup_;
}

template <typename D, typename... T>
template <std::size_t... Idx>
typename ZiperatorBase<D, T...>::value_type ZiperatorBase<D, T...>::dereference_(
  std::index_sequence<Idx...>) const
{
  return {*std::get<Idx>(derived_().iters_())...};
}

template <typename D, typename... T>
template <std::size_t... Idx>
void ZiperatorBase<D, T...>::increment_(std::index_sequence<Idx...>)
{
  (++std::get<Idx>(derived_().iters_()), ...);
}

template <typename D, typename... T>
template <std::size_t... Idx>
void ZiperatorBase<D, T...>::pluseq_(std::index_sequence<Idx...>, difference_type n)
{
  ((std::get<Idx>(derived_().iters_()) += n), ...);
}

template <typename D, typename... T>
template <std::size_t... Idx>
void ZiperatorBase<D, T...>::decrement_(std::index_sequence<Idx...>)
{
  (--std::get<Idx>(derived_().iters_()), ...);
}

template <typename D, typename... T>
template <std::size_t... Idx>
void ZiperatorBase<D, T...>::minuseq_(std::index_sequence<Idx...>, difference_type n)
{
  ((std::get<Idx>(derived_().iters_()) -= n), ...);
}

template <typename D, typename... T>
template <std::size_t... Idx>
bool ZiperatorBase<D, T...>::lessthan_(std::index_sequence<Idx...>,
                                       const ZiperatorBase& other) const
{
  return ((std::get<Idx>(derived_().iters_()) < std::get<Idx>(other.derived_().iters_())) && ...);
}

template <typename D, typename... T>
template <std::size_t... Idx>
bool ZiperatorBase<D, T...>::eq_(std::index_sequence<Idx...>, const ZiperatorBase& other) const
{
  return ((std::get<Idx>(derived_().iters_()) == std::get<Idx>(other.derived_().iters_())) && ...);
}

// ==========================================================================================

template <typename... T>
template <std::size_t... Idx>
bool ZiperatorShortest<T...>::eq_(std::index_sequence<Idx...>, const ZiperatorShortest& other) const
{
  // This is safe to use (and results in "shortest zip" semantics) when only comparing
  // ZipIterators that satisfy one of these invariants:
  // * it was derived by advancing the begin() ZipIterator, and thus has all its sub-iterators on
  //   the same index
  // * it is the end() ZipIterator, and the user guarantees never to advance a ZipIterator once it
  //   compares equal to end()
  return ((std::get<Idx>(this->derived_().iters_()) == std::get<Idx>(other.derived_().iters_())) ||
          ...);
}

// ==========================================================================================

template <typename... T>
template <std::size_t... Idx>
bool ZiperatorEqual<T...>::eq_(std::index_sequence<Idx...>, const ZiperatorEqual& other) const
{
  // This is safe to use (and results in "equal zip" semantics) when only comparing
  // ZipIterators that satisfy one of these invariants:
  // * it was derived by advancing the begin() ZipIterator, and thus has all its sub-iterators on
  //   the same index
  // * it is the end() ZipIterator, and the user guarantees never to advance a ZipIterator once it
  //   compares equal to end()
  return std::get<0>(this->derived_().iters_()) == std::get<0>(other.derived_().iters_());
}

// ==========================================================================================

template <template <typename...> class ZIT, typename... T>
Zipper<ZIT, T...>::Zipper(T&&... objs) : objs_{std::forward<T>(objs)...}
{
}

template <template <typename...> class ZIT, typename... T>
typename Zipper<ZIT, T...>::iterator Zipper<ZIT, T...>::begin()
{
  return begin_(sequence{});
}

template <template <typename...> class ZIT, typename... T>
typename Zipper<ZIT, T...>::const_iterator Zipper<ZIT, T...>::cbegin() const
{
  return begin_(sequence{});
}

template <template <typename...> class ZIT, typename... T>
typename Zipper<ZIT, T...>::const_iterator Zipper<ZIT, T...>::begin() const
{
  return cbegin();
}

template <template <typename...> class ZIT, typename... T>
typename Zipper<ZIT, T...>::iterator Zipper<ZIT, T...>::end()
{
  return end_(sequence{});
}

template <template <typename...> class ZIT, typename... T>
typename Zipper<ZIT, T...>::const_iterator Zipper<ZIT, T...>::cend() const
{
  return end_(sequence{});
}

template <template <typename...> class ZIT, typename... T>
typename Zipper<ZIT, T...>::const_iterator Zipper<ZIT, T...>::end() const
{
  return cend();
}

// ==========================================================================================

template <template <typename...> class ZIT, typename... T>
template <std::size_t... Ns>
typename Zipper<ZIT, T...>::iterator Zipper<ZIT, T...>::begin_(std::index_sequence<Ns...>)
{
  return iterator{std::begin(std::get<Ns>(objs_))...};
}

template <template <typename...> class ZIT, typename... T>
template <std::size_t... Ns>
typename Zipper<ZIT, T...>::const_iterator Zipper<ZIT, T...>::begin_(
  std::index_sequence<Ns...>) const
{
  return const_iterator{std::cbegin(std::get<Ns>(objs_))...};
}

template <template <typename...> class ZIT, typename... T>
template <std::size_t... Ns>
typename Zipper<ZIT, T...>::iterator Zipper<ZIT, T...>::end_(std::index_sequence<Ns...>)
{
  return iterator{std::end(std::get<Ns>(objs_))...};
}

template <template <typename...> class ZIT, typename... T>
template <std::size_t... Ns>
typename Zipper<ZIT, T...>::const_iterator Zipper<ZIT, T...>::end_(std::index_sequence<Ns...>) const
{
  return const_iterator{std::cend(std::get<Ns>(objs_))...};
}

}  // namespace zip_detail

template <typename... T>
zip_detail::Zipper<zip_detail::ZiperatorShortest, T...> zip_shortest(T&&... args)
{
  return {std::forward<T>(args)...};
}

namespace zip_detail {

template <typename T>
using has_size = decltype(std::size(std::declval<T>()));

[[noreturn]] void throw_unequal_container_sizes();

}  // namespace zip_detail

template <typename... T>
zip_detail::Zipper<zip_detail::ZiperatorEqual, T...> zip_equal(T&&... args)
{
  if constexpr (LEGATE_DEFINED(LEGATE_USE_DEBUG) && (sizeof...(args) > 1) &&
                std::conjunction_v<detail::is_detected<zip_detail::has_size, T>...>) {
    const auto all_same_size = [](const auto& a0, const auto&... rest) {
      return ((std::size(a0) == std::size(rest)) && ...);
    }(args...);

    if (!all_same_size) {
      zip_detail::throw_unequal_container_sizes();
    }
  }
  return {std::forward<T>(args)...};
}

}  // namespace legate::detail
