/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include "config.hpp"
#include "meta.hpp"
#include "type_traits.hpp"

#if LEGATE_STL_HAS_STD_RANGES()

#include <ranges>

// Include this last
#include "prefix.hpp"

namespace legate::stl {
using std::ranges::begin;
using std::ranges::end;
using std::ranges::range;

using std::ranges::iterator_t;
using std::ranges::range_reference_t;
using std::ranges::range_value_t;
using std::ranges::sentinel_t;
}  // namespace legate::stl

#else

#include <iterator>

// Include this last
#include "prefix.hpp"

namespace legate::stl {

namespace detail {
namespace begin {
void begin();

template <typename Ty>
using member_begin_t = decltype(std::declval<Ty>().begin());

template <typename Ty>
using free_begin_t = decltype(begin(std::declval<Ty>()));

template <typename Ty>
using begin_fn = meta::if_c<meta::evaluable_q<member_begin_t, Ty>,
                            meta::quote<member_begin_t>,
                            meta::quote<free_begin_t>>;

template <typename Ty>
using begin_result_t = meta::eval<begin_fn<Ty>, Ty>;

struct tag {
  template <typename Range>
  static constexpr bool _nothrow_begin() noexcept
  {
    if constexpr (meta::evaluable_q<member_begin_t, Range>) {
      return noexcept(std::declval<Range>().begin());
    } else {
      return noexcept(begin(std::declval<Range>()));
    }
  }

  template <typename Range>
  auto operator()(Range&& rng) const noexcept(_nothrow_begin<Range>()) -> begin_result_t<Range>
  {
    if constexpr (meta::evaluable_q<member_begin_t, Range>) {
      return ((Range&&)rng).begin();
    } else {
      return begin(((Range&&)rng));
    }
  }
};
}  // namespace begin

namespace end {
void end();

template <typename Ty>
using member_end_t = decltype(std::declval<Ty>().end());

template <typename Ty>
using free_end_t = decltype(end(std::declval<Ty>()));

template <typename Ty>
using end_fn = meta::
  if_c<meta::evaluable_q<member_end_t, Ty>, meta::quote<member_end_t>, meta::quote<free_end_t>>;

template <typename Ty>
using end_result_t = meta::eval<end_fn<Ty>, Ty>;

struct tag {
  template <typename Range>
  static constexpr bool _nothrow_end() noexcept
  {
    if constexpr (meta::evaluable_q<member_end_t, Range>) {
      return noexcept(std::declval<Range>().end());
    } else {
      return noexcept(end(std::declval<Range>()));
    }
  }

  template <typename Range>
  auto operator()(Range&& rng) const noexcept(_nothrow_end<Range>()) -> end_result_t<Range>
  {
    if constexpr (meta::evaluable_q<member_end_t, Range>) {
      return ((Range&&)rng).end();
    } else {
      return end(((Range&&)rng));
    }
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
template <typename Range>
using iterator_t = decltype(stl::begin((std::declval<Range>())));

template <typename Range>
using sentinel_t = decltype(stl::end((std::declval<Range>())));

template <typename Range>
using range_reference_t = decltype(*stl::begin((std::declval<Range>())));

template <typename Range>
using range_value_t = typename std::iterator_traits<iterator_t<Range>>::value_type;

//////////////////////////////////////////////////////////////////////////////////////////////////
namespace detail {
template <typename Iter>
auto is_iterator_like_(Iter iter) -> decltype((void)++iter, (void)*iter, (void)(iter == iter));

template <typename Range>
auto is_range_like_(Range&& rng)
  -> decltype(detail::is_iterator_like_(stl::begin(rng)), (void)(stl::begin(rng) == stl::end(rng)));

template <typename Range>
using is_range_like_t = decltype(detail::is_range_like_(std::declval<Range>()));
}  // namespace detail

template <typename Range>
inline constexpr bool range = meta::evaluable_q<detail::is_range_like_t, Range>;
}  // namespace legate::stl
#endif

namespace legate::stl {
//////////////////////////////////////////////////////////////////////////////////////////////////
namespace tags {
namespace as_range {
void as_range();

template <typename T>
using as_range_t = decltype(as_range(std::declval<T>()));

template <typename T>
inline constexpr bool range_like_ = meta::evaluable_q<as_range_t, T>;

template <typename T>
using as_range_result_t =
  meta::eval<meta::if_c<range<T>, meta::always<T>, meta::quote<as_range_t>>, T>;

struct tag {
  template <typename T>
  static constexpr bool _noexcept_as_range() noexcept
  {
    if constexpr (range<T>) {
      return noexcept(std::decay_t<T>{std::declval<T>()});
    } else {
      return noexcept(as_range(std::declval<T>()));
    }
  }

  template <typename T>
  as_range_result_t<T> operator()(T&& rng) const noexcept(_noexcept_as_range<T>())
  {
    if constexpr (range<T>) {
      return (T&&)rng;
    } else {
      return as_range((T&&)rng);
    }
  }
};
}  // namespace as_range

inline namespace obj {
inline constexpr as_range::tag as_range{};
}  // namespace obj
}  // namespace tags

using namespace tags::obj;

template <typename T>
using as_range_t = call_result_c_t<as_range, T>;

}  // namespace legate::stl

#include "suffix.hpp"
