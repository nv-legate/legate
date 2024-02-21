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

#include "meta.hpp"

#include <type_traits>

// Include this last:
#include "prefix.hpp"

namespace legate::stl {
namespace detail {
template <class T>
T&& declval() noexcept;
}  // namespace detail

#if __cpp_lib_remove_cvref >= 20171L
using std::remove_cvref_t;
#else
template <class T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;
#endif

template <class A, class B>
inline constexpr bool same_as = false;

template <class A>
inline constexpr bool same_as<A, A> = true;

template <const auto& Value>
using typeof_t = remove_cvref_t<decltype(Value)>;

template <class Fun, class... Args>
using call_result_t = decltype(detail::declval<Fun>()(detail::declval<Args>()...));

template <const auto& Fun, class... Args>
using call_result_c_t = call_result_t<typeof_t<Fun>, Args...>;

template <class Fun, class... Args>
inline constexpr bool callable = meta::evaluable_q<call_result_t, Fun, Args...>;

template <const auto& Fun, class... Args>
inline constexpr bool callable_c = meta::evaluable_q<call_result_t, typeof_t<Fun>, Args...>;

template <class Type>
using observer_ptr = Type*;

template <bool MakeConst, class Type>
using const_if_t = meta::if_c<MakeConst, const Type, Type>;

}  // namespace legate::stl

#include "suffix.hpp"
