/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/experimental/stl/detail/config.hpp>
#include <legate/experimental/stl/detail/utility.hpp>
#include <legate/utilities/assert.h>

#include <functional>
#include <tuple>

// Include this last:
#include <legate/experimental/stl/detail/prefix.hpp>

namespace legate::experimental::stl {

//////////////////////////////////////////////////////////////////////////////////////////////////
namespace detail {

template <typename Fun, std::size_t... Is>
[[nodiscard]] constexpr auto with_indices_impl_1(Fun&& fun, std::index_sequence<Is...>)
  -> decltype(std::forward<Fun>(fun)(Is...))
{
  return std::forward<Fun>(fun)(Is...);
}

template <template <std::size_t> typename Tfx, typename Fun, std::size_t... Is>
[[nodiscard]] constexpr auto with_indices_impl_2(Fun&& fun, std::index_sequence<Is...>)
  -> decltype(std::forward<Fun>(fun)(Tfx<Is>()()...))
{
  return std::forward<Fun>(fun)(Tfx<Is>()()...);
}

}  // namespace detail

template <std::size_t N, typename Fun>
[[nodiscard]] constexpr auto with_indices(Fun&& fun)
  -> decltype(detail::with_indices_impl_1(std::forward<Fun>(fun), std::make_index_sequence<N>()))
{
  return detail::with_indices_impl_1(std::forward<Fun>(fun), std::make_index_sequence<N>());
}

template <std::size_t N, template <std::size_t> typename Tfx, typename Fun>
[[nodiscard]] constexpr auto with_indices(Fun&& fun)
  -> decltype(detail::with_indices_impl_2<Tfx>(std::forward<Fun>(fun),
                                               std::make_index_sequence<N>()))
{
  return detail::with_indices_impl_2<Tfx>(std::forward<Fun>(fun), std::make_index_sequence<N>());
}

/// \cond
namespace detail {

template <typename Fn, typename... Args>
class BinderBack {
 public:
  Fn fn_{};
  std::tuple<Args...> args_{};

  template <typename... Ts>
    requires(std::is_invocable_v<Fn, Ts..., Args...>)
  LEGATE_HOST_DEVICE decltype(auto) operator()(Ts&&... params)
  {
    return std::apply([&](auto&... args) { return fn_(std::forward<Ts>(params)..., args...); },
                      args_);
  }
};

template <typename Fn, typename... Args>
BinderBack(Fn, Args...) -> BinderBack<Fn, Args...>;

}  // namespace detail
/// \endcond

//////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Fn, typename... Args>
LEGATE_HOST_DEVICE [[nodiscard]] auto bind_back(Fn fn, Args&&... args)
{
  if constexpr (sizeof...(args) == 0) {
    return fn;
  } else {
    return detail::BinderBack{std::move(fn), std::forward<Args>(args)...};
  }
  LEGATE_UNREACHABLE();
}

//////////////////////////////////////////////////////////////////////////////////////////////////
namespace detail {

template <typename Function, typename... Ignore>
class DropNArgs {
 public:
  Function fun_{};

  template <typename... Args>
  [[nodiscard]] constexpr decltype(auto) operator()(Ignore..., Args&&... args) const
  {
    return fun_(std::forward<Args>(args)...);
  }
};

}  // namespace detail

template <std::size_t Count, typename Function>
using drop_n_args =
  meta::fill_n<Count, ignore, meta::bind_front<meta::quote<detail::DropNArgs>, Function>>;

template <std::size_t Count, typename Function>
[[nodiscard]] drop_n_args<Count, std::decay_t<Function>> drop_n_fn(Function&& fun)
{
  return {std::forward<Function>(fun)};
}

}  // namespace legate::experimental::stl

#include <legate/experimental/stl/detail/suffix.hpp>
