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

#include <cstdint>
#include <utility>

// Include this last:
#include "prefix.hpp"

namespace legate::stl {
namespace meta {
struct na;

struct empty {};

namespace detail {
template <typename T>
struct type_ {
  using type = T;
};
}  // namespace detail

template <typename T>
using type = detail::type_<T>;

template <typename T>
using identity = typename type<T>::type;

template <auto Value>
using constant = std::integral_constant<decltype(Value), Value>;

namespace detail {
template <std::size_t>
struct eval_ {
  template <template <typename...> typename Fun, typename... Args>
  using eval = Fun<Args...>;
};
}  // namespace detail

template <template <typename...> typename Fun, typename... Args>
using eval_q = typename detail::eval_<sizeof...(Args)>::template eval<Fun, Args...>;

template <typename Fun, typename... Args>
using eval = eval_q<Fun::template eval, Args...>;

template <typename... Ts>
struct list {
  template <typename Fn>
  using eval = meta::eval<Fn, Ts...>;
};

template <template <typename...> typename Fun>
struct quote {
  template <typename... Args>
  using eval = eval_q<Fun, Args...>;
};

template <typename A, typename B>
using first_t = A;

template <typename A, typename B>
using second_t = B;

template <typename A>
using void_t = second_t<A, void>;

namespace detail {
template <template <typename...> typename, typename...>
struct test_evaluable_with_;

struct test_evaluable_with_base_ {
  template <template <typename...> typename C, typename... Args>
  friend constexpr first_t<bool, C<Args...>> test_evaluable_(test_evaluable_with_<C, Args...>*)
  {
    return true;
  }
};

template <template <typename...> typename C, typename... Args>
struct test_evaluable_with_ : test_evaluable_with_base_ {};

constexpr bool test_evaluable_(...) { return false; }

template <template <typename...> typename Fun, typename... Args>
inline constexpr bool evaluable_q =
  test_evaluable_(static_cast<test_evaluable_with_<Fun, Args...>*>(nullptr));
}  // namespace detail

using detail::evaluable_q;

template <typename Fun, typename... Args>
inline constexpr bool evaluable = evaluable_q<Fun::template eval, Args...>;

namespace detail {
template <bool>
struct if_ {
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
struct always {
  template <typename...>
  using eval = T;
};

template <template <typename...> typename Fun, typename Default>
struct quote_or {
  template <bool Evaluable>
  struct maybe : if_c<Evaluable, quote<Fun>, always<Default>> {};

  template <typename... Args>
  using maybe_t = maybe<evaluable_q<Fun, Args...>>;

  template <typename... Args>
  using eval = eval<maybe_t<Args...>, Args...>;
};

template <typename Fun, typename... Args>
struct bind_front {
  template <typename... OtherArgs>
  using eval = eval<Fun, Args..., OtherArgs...>;
};

template <typename Fun, typename... Args>
struct bind_back {
  template <typename... OtherArgs>
  using eval = eval<Fun, OtherArgs..., Args...>;
};

namespace detail {
template <typename Head, typename... Tail>
using front_ = Head;
}  // namespace detail

template <typename... Ts>
using front = eval_q<detail::front_, Ts...>;

namespace detail {
template <typename Indices>
struct fill_n_;

template <std::size_t... Is>
struct fill_n_<std::index_sequence<Is...>> {
  template <typename Value, typename Continuation>
  using eval = eval<Continuation, first_t<Value, constant<Is>>...>;
};
}  // namespace detail

template <std::size_t Count, typename Value, typename Continuation = quote<list>>
using fill_n = eval<detail::fill_n_<std::make_index_sequence<Count>>, Value, Continuation>;

}  // namespace meta
}  // namespace legate::stl

#include "suffix.hpp"
