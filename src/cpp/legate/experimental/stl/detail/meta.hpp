/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/experimental/stl/detail/config.hpp>
#include <legate/utilities/detail/type_traits.h>

#include <cstdint>
#include <utility>

// Include this last:
#include <legate/experimental/stl/detail/prefix.hpp>

/**
 * @cond
 */

namespace legate::experimental::stl::meta {

struct na;

struct empty {};  // NOLINT(readability-identifier-naming)

template <auto Value>
using constant = std::integral_constant<decltype(Value), Value>;

namespace detail {

template <std::size_t>
struct eval_ {  // NOLINT(readability-identifier-naming)
  template <template <typename...> typename Fun, typename... Args>
  using eval = Fun<Args...>;
};

}  // namespace detail

template <template <typename...> typename Fun, typename... Args>
using eval_q = typename detail::eval_<sizeof...(Args)>::template eval<Fun, Args...>;

template <typename Fun, typename... Args>
using eval = eval_q<Fun::template eval, Args...>;

template <typename... Ts>
struct list {  // NOLINT(readability-identifier-naming)
  template <typename Fn>
  using eval = meta::eval<Fn, Ts...>;
};

template <template <typename...> typename Fun>
struct quote {  // NOLINT(readability-identifier-naming)
  template <typename... Args>
  using eval = eval_q<Fun, Args...>;
};

namespace detail {

template <typename Head, typename... Tail>
using front_ = Head;

template <template <typename...> typename, typename...>
struct test_evaluable_with;  // NOLINT(readability-identifier-naming)

struct test_evaluable_with_base {  // NOLINT(readability-identifier-naming)
  template <template <typename...> typename C, typename... Args>
  friend constexpr front_<bool, C<Args...>> test_evaluable(test_evaluable_with<C, Args...>*)
  {
    return true;
  }
};

template <template <typename...> typename C, typename... Args>
struct test_evaluable_with : test_evaluable_with_base {};

constexpr bool test_evaluable(...) { return false; }

template <template <typename...> typename Fun, typename... Args>
inline constexpr bool evaluable_q =  // NOLINT(readability-identifier-naming)
  test_evaluable(static_cast<test_evaluable_with<Fun, Args...>*>(nullptr));

}  // namespace detail

using detail::evaluable_q;

template <typename Fun, typename... Args>
inline constexpr bool evaluable =  // NOLINT(readability-identifier-naming)
  evaluable_q<Fun::template eval, Args...>;

namespace detail {

template <bool>
struct if_ {  // NOLINT(readability-identifier-naming)
  template <typename Then, typename... Else>
  using eval = Then;
};

template <>
struct if_<false> {
  template <typename Then, typename Else>
  using eval = Else;
};

}  // namespace detail

template <bool Cond, typename Then = void, typename... Else>
using if_c = eval<detail::if_<Cond>, Then, Else...>;

template <typename T>
struct always {  // NOLINT(readability-identifier-naming)
  template <typename...>
  using eval = T;
};

template <template <typename...> typename Fun, typename Default>
struct quote_or {  // NOLINT(readability-identifier-naming)
  template <bool Evaluable>
  struct maybe  // NOLINT(readability-identifier-naming)
    : if_c<Evaluable, quote<Fun>, always<Default>> {};

  template <typename... Args>
  using maybe_t = maybe<evaluable_q<Fun, Args...>>;

  template <typename... Args>
  using eval = eval<maybe_t<Args...>, Args...>;
};

template <typename Fun, typename... Args>
struct bind_front {  // NOLINT(readability-identifier-naming)
  template <typename... OtherArgs>
  using eval = eval<Fun, Args..., OtherArgs...>;
};

template <typename Fun, typename... Args>
struct bind_back {  // NOLINT(readability-identifier-naming)
  template <typename... OtherArgs>
  using eval = eval<Fun, OtherArgs..., Args...>;
};

template <typename... Ts>
using front = eval_q<detail::front_, Ts...>;

namespace detail {

template <typename Indices>
struct fill_n_;

template <std::size_t... Is>
struct fill_n_<std::index_sequence<Is...>> {
  template <typename Value, typename Continuation>
  using eval = eval<Continuation, front_<Value, constant<Is>>...>;
};

}  // namespace detail

template <std::size_t Count, typename Value, typename Continuation = quote<list>>
using fill_n = eval<detail::fill_n_<std::make_index_sequence<Count>>, Value, Continuation>;

}  // namespace legate::experimental::stl::meta

/**
 * @endcond
 */

#include <legate/experimental/stl/detail/suffix.hpp>
