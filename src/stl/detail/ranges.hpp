/*
 * Copyright (c) 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License Version 2.0 with LLVM Exceptions
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "config.hpp"
#include "type_traits.hpp"

#if LEGATE_STL_HAS_STD_RANGES()

#include <ranges>

namespace legate::stl {
using std::ranges::begin;
using std::ranges::end;

using std::ranges::iterator_t;
using std::ranges::range_reference_t;
using std::ranges::range_value_t;
using std::ranges::sentinel_t;
}  // namespace legate::stl

#else

#include "prefix.hpp"

#include <iterator>

namespace legate::stl {

namespace detail {
namespace begin {
void begin();

template <class Ty>
using member_begin_t = decltype(std::declval<Ty>().begin());

template <class Ty>
using free_begin_t = decltype(begin(std::declval<Ty>()));

struct tag {
  template(class Range)
    requires(meta::evaluable_q<member_begin_t, Range>)
  auto operator()(Range&& rng) const noexcept(noexcept(((Range&&)rng).begin()))
    -> member_begin_t<Range>
  {
    return ((Range&&)rng).begin();
  }

  template(class Range)
    requires(meta::evaluable_q<free_begin_t, Range> && !meta::evaluable_q<member_begin_t, Range>)
  auto operator()(Range&& rng) const noexcept(noexcept(begin(((Range&&)rng))))
    -> free_begin_t<Range>
  {
    return begin(((Range&&)rng));
  }
};
}  // namespace begin

namespace end {
void end();

template <class Ty>
using member_end_t = decltype(std::declval<Ty>().end());

template <class Ty>
using free_end_t = decltype(end(std::declval<Ty>()));

struct tag {
  template(class Range)
    requires(meta::evaluable_q<member_end_t, Range>)
  auto operator()(Range&& rng) const noexcept(noexcept(((Range&&)rng).end())) -> member_end_t<Range>
  {
    return ((Range&&)rng).end();
  }

  template(class Range)
    requires(meta::evaluable_q<free_end_t, Range> && !meta::evaluable_q<member_end_t, Range>)
  auto operator()(Range&& rng) const noexcept(noexcept(end(((Range&&)rng)))) -> free_end_t<Range>
  {
    return end(((Range&&)rng));
  }
};
}  // namespace end
}  // namespace detail

namespace tag {
inline constexpr detail::begin::tag begin{};
inline constexpr detail::end::tag end{};
}  // namespace tag

using namespace tag;

//////////////////////////////////////////////////////////////////////////////////////////////////
template <class Range>
using iterator_t = decltype(stl::begin((std::declval<Range>())));

template <class Range>
using sentinel_t = decltype(stl::end((std::declval<Range>())));

template <class Range>
using range_reference_t = decltype(*stl::begin((std::declval<Range>())));

template <class Range>
using range_value_t = typename std::iterator_traits<iterator_t<Range>>::value_type;

//////////////////////////////////////////////////////////////////////////////////////////////////
namespace detail {
template <class Iter>
auto is_iterator_like_(Iter iter) -> decltype((void)++iter, (void)*iter, (void)(iter == iter));

template <class Range>
auto is_range_like_(Range&& rng)
  -> decltype(detail::is_iterator_like_(stl::begin(rng)), (void)(stl::begin(rng) == stl::end(rng)));

template <class Range>
using is_range_like_t = decltype(detail::is_range_like_(std::declval<Range>()));
}  // namespace detail

template <class Range>
inline constexpr bool range = meta::evaluable_q<detail::is_range_like_t, Range>;

//////////////////////////////////////////////////////////////////////////////////////////////////
namespace tags {
namespace as_range {
void as_range();

template <class T>
using as_range_t = decltype(as_range(std::declval<T>()));

template <class T>
inline constexpr bool range_like_ = meta::evaluable_q<as_range_t, T>;

struct tag {
  template(class T)
    requires(range<T>)
  T operator()(T&& rng) const noexcept
  {
    return (T&&)rng;
  }

  template(class T)
    requires(range_like_<T> && !range<T>)
  as_range_t<T> operator()(T&& rng) const
  {
    return as_range((T&&)rng);  // This call is intentionally unqualified
  }
};
}  // namespace as_range

inline namespace obj {
inline constexpr as_range::tag as_range{};
}  // namespace obj
}  // namespace tags

using namespace tags::obj;

template <class T>
using as_range_t = call_result_c_t<as_range, T>;

}  // namespace legate::stl

#include "suffix.hpp"

#endif
