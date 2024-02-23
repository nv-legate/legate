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

#include "core/utilities/assert.h"

#include "config.hpp"
#include "utility.hpp"

#include <functional>
#include <tuple>

// Include this last:
#include "prefix.hpp"

namespace legate::stl {

//////////////////////////////////////////////////////////////////////////////////////////////////
namespace detail {
template <class Fun, std::size_t... Is>
constexpr auto with_indices_impl_1(Fun&& fun, std::index_sequence<Is...>)
  -> decltype(std::forward<Fun>(fun)(Is...))
{
  return std::forward<Fun>(fun)(Is...);
}

template <template <std::size_t> class Tfx, class Fun, std::size_t... Is>
constexpr auto with_indices_impl_2(Fun&& fun, std::index_sequence<Is...>)
  -> decltype(std::forward<Fun>(fun)(Tfx<Is>()()...))
{
  return std::forward<Fun>(fun)(Tfx<Is>()()...);
}

template <std::size_t N>
struct index_t {
  std::integral_constant<std::size_t, N> operator()() const { return {}; }
};
}  // namespace detail

template <std::size_t N, class Fun>
constexpr auto with_indices(Fun&& fun)
  -> decltype(detail::with_indices_impl_1(std::forward<Fun>(fun), std::make_index_sequence<N>()))
{
  return detail::with_indices_impl_1(std::forward<Fun>(fun), std::make_index_sequence<N>());
}

template <std::size_t N, template <std::size_t> class Tfx, class Fun>
constexpr auto with_indices(Fun&& fun)
  -> decltype(detail::with_indices_impl_2<Tfx>(std::forward<Fun>(fun),
                                               std::make_index_sequence<N>()))
{
  return detail::with_indices_impl_2<Tfx>(std::forward<Fun>(fun), std::make_index_sequence<N>());
}

/// \cond
namespace detail {
template <class Fn, class... Args>
struct binder_back {
  Fn fn_;
  std::tuple<Args...> args_;

  template <class... Ts>
    requires(std::is_invocable_v<Fn, Ts..., Args...>)
  LEGATE_STL_ATTRIBUTE((host, device)) decltype(auto) operator()(Ts&&... params)
  {
    return std::apply([&](auto&... args) { return fn_(std::forward<Ts>(params)..., args...); },
                      args_);
  }
};

template <class Fn, class... Args>
binder_back(Fn, Args...) -> binder_back<Fn, Args...>;
}  // namespace detail
/// \endcond

//////////////////////////////////////////////////////////////////////////////////////////////////
template <class Fn, class... Args>
LEGATE_STL_ATTRIBUTE((host, device))
auto bind_back(Fn fn, Args&&... args)
{
  if constexpr (sizeof...(args) == 0) {
    return fn;
  } else {
    return detail::binder_back{std::move(fn), std::forward<Args>(args)...};
  }
  LegateUnreachable();
}

//////////////////////////////////////////////////////////////////////////////////////////////////
namespace detail {
template <class Function, class... Ignore>
struct drop_n_args {
  Function fun_;

  template <class... Args>
  constexpr decltype(auto) operator()(Ignore..., Args&&... args) const
  {
    return fun_(std::forward<Args>(args)...);
  }
};
}  // namespace detail

template <std::size_t Count, class Function>
using drop_n_args =
  meta::fill_n<Count, ignore, meta::bind_front<meta::quote<detail::drop_n_args>, Function>>;

template <std::size_t Count, class Function>
drop_n_args<Count, Function> drop_n_fn(Function fun)
{
  return {std::move(fun)};
}

}  // namespace legate::stl

#include "suffix.hpp"
