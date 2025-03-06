/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/cpp_version.h>

#include <type_traits>

namespace legate::detail {

namespace move_detail {

template <typename T>
struct MoveConversionSFINAEHelper {
  // The point here is that these are explicit
  // NOLINTBEGIN(google-explicit-constructor)
  operator T const&();
  operator T&&();
  // NOLINTEND(google-explicit-constructor)
};

}  // namespace move_detail

// A useful helper to detect if a class is TRULY move-constructible. It returns true if (and
// only if) the class has a move ctor, i.e. Foo f{std::move(another_foo)} results in
// Foo(Foo &&) being called.
//
// One cannot simply use std::is_move_constructible to determine whether std::move(something)
// will actually result in a move-constructor being called since:
//
// "Types without a move constructor, but with a copy constructor that accepts const T&
// arguments, satisfy std::is_move_constructible."
//
// So we have this nifty helper which abuses operator overloading. If a particular function has
// 2 equally viable overload candidate, it fails to compile. The sfinae helper has
// both an implicit move and const-ref (i.e. copy) operator, so if the class has BOTH a copy
// and move ctor there will exist 2 equally viable constructors, and the SFINAE kicks in.
//
// So this
template <typename T>
struct is_pure_move_constructible
  : std::integral_constant<
      bool,
      std::is_move_constructible_v<T> &&
        !std::is_constructible_v<T, move_detail::MoveConversionSFINAEHelper<T>>> {};

template <typename T>
inline constexpr bool is_pure_move_constructible_v = is_pure_move_constructible<T>::value;

#ifndef __NVCC__
// This does not appear to work for NVCC...
namespace is_pure_move_constructible_test {

struct MoveConstructible {
  MoveConstructible(MoveConstructible&&) = default;
};

struct CopyConstructible {
  CopyConstructible(const CopyConstructible&) = default;
};

struct CopyAndMoveConstructible {
  CopyAndMoveConstructible(const CopyAndMoveConstructible&) = default;
  CopyAndMoveConstructible(CopyAndMoveConstructible&&)      = default;
};

static_assert(!is_pure_move_constructible_v<CopyConstructible>);
static_assert(is_pure_move_constructible_v<MoveConstructible>);
static_assert(is_pure_move_constructible_v<CopyAndMoveConstructible>);

}  // namespace is_pure_move_constructible_test
#endif

// Same as is_pure_move_constructible, but for operator=(Foo &&).
template <typename T>
struct is_pure_move_assignable
  : std::integral_constant<bool,
                           std::is_move_assignable_v<T> &&
                             !std::is_assignable_v<T, move_detail::MoveConversionSFINAEHelper<T>>> {
};

template <typename T>
inline constexpr bool is_pure_move_assignable_v = is_pure_move_assignable<T>::value;

#ifndef __NVCC__
// This does not appear to work for NVCC...
namespace is_pure_move_assignable_test {

struct MoveAssignable {
  MoveAssignable& operator=(MoveAssignable&&) = default;
};

struct CopyAssignable {
  CopyAssignable& operator=(const CopyAssignable&) = default;
};

struct CopyAndMoveAssignable {
  CopyAndMoveAssignable& operator=(const CopyAndMoveAssignable&) = default;
  CopyAndMoveAssignable& operator=(CopyAndMoveAssignable&&)      = default;
};

static_assert(!is_pure_move_assignable_v<CopyAssignable>);
static_assert(is_pure_move_assignable_v<MoveAssignable>);
static_assert(is_pure_move_assignable_v<CopyAndMoveAssignable>);

}  // namespace is_pure_move_assignable_test
#endif

template <typename From, typename To>
struct is_ptr_compat : std::is_convertible<From*, To*> {};

template <typename From, typename To>
inline constexpr bool is_ptr_compat_v = is_ptr_compat<From, To>::value;

template <typename T, typename... Ts>
struct is_same_as_one_of : std::bool_constant<(... || std::is_same<T, Ts>{})> {};

template <typename T, typename... Ts>
inline constexpr bool is_same_as_one_of_v = is_same_as_one_of<T, Ts...>::value;

template <typename T>
[[nodiscard]] constexpr std::underlying_type_t<T> to_underlying(T e) noexcept
{
  static_assert(std::is_enum_v<T>);
  return static_cast<std::underlying_type_t<T>>(e);
}

namespace detected_detail {

template <typename Default,
          typename AlwaysVoid,
          template <typename...> typename Op,
          typename... Args>
struct detector : std::false_type {  // NOLINT(readability-identifier-naming)
  using type = Default;
};

template <typename Default, template <typename...> typename Op, typename... Args>
struct detector<Default, std::void_t<Op<Args...>>, Op, Args...> : std::true_type {
  using type = Op<Args...>;
};

struct nonesuch {  // NOLINT(readability-identifier-naming)
  ~nonesuch()                     = delete;
  nonesuch(nonesuch const&)       = delete;
  void operator=(nonesuch const&) = delete;
};

}  // namespace detected_detail

template <typename Default, template <typename...> typename Op, typename... Args>
using detected_or = detected_detail::detector<Default, void, Op, Args...>;

template <template <typename...> typename Op, typename... Args>
using is_detected = detected_or<detected_detail::nonesuch, Op, Args...>;

template <template <typename...> typename Op, typename... Args>
inline constexpr bool is_detected_v = is_detected<Op, Args...>::value;

template <template <typename...> class Op, typename... Args>
using is_detected_t = typename is_detected<Op, Args...>::type;

LEGATE_CPP_VERSION_TODO(20, "use std::type_identity");

template <typename T>
struct type_identity {  // NOLINT(readability-identifier-naming)
  using type = T;
};

template <typename T>
using type_identity_t = typename type_identity<T>::type;

template <typename... T>
class Overload : public T... {
 public:
  using T::operator()...;
};

template <typename... T>
Overload(T...) -> Overload<T...>;

}  // namespace legate::detail
