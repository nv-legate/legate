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

#include <type_traits>
#include <utility>

namespace legate {

namespace detail {

template <bool t_empty, bool u_empty>
struct compressed_pair_selector;

template <>
struct compressed_pair_selector<false, false> : std::integral_constant<int, 0> {};

template <>
struct compressed_pair_selector<true, false> : std::integral_constant<int, 1> {};

template <>
struct compressed_pair_selector<false, true> : std::integral_constant<int, 2> {};

template <>
struct compressed_pair_selector<true, true> : std::integral_constant<int, 3> {};

template <typename T, typename U, int selector>
class compressed_pair_impl;

// selector = 0, neither are empty, derive directly from std::pair
template <typename T, typename U>
class compressed_pair_impl<T, U, 0> : std::pair<T, U> {
  using base_type = std::pair<T, U>;

 public:
  using base_type::base_type;
  using typename base_type::first_type;
  using typename base_type::second_type;

  [[nodiscard]] first_type& first() noexcept { return static_cast<base_type&>(*this).first; }
  [[nodiscard]] const first_type& first() const noexcept
  {
    return static_cast<const base_type&>(*this).first;
  }

  [[nodiscard]] second_type& second() noexcept { return static_cast<base_type&>(*this).second; }
  [[nodiscard]] const second_type& second() const noexcept
  {
    return static_cast<const base_type&>(*this).second;
  }
};

// selector = 1, T is empty
template <typename T, typename U>
class compressed_pair_impl<T, U, 1> : T {
  using base_type = T;

 public:
  using base_type::base_type;
  using first_type  = T;
  using second_type = U;

  compressed_pair_impl() = default;

  compressed_pair_impl(first_type x, second_type y) : base_type{std::move(x)}, second_{std::move(y)}
  {
  }

  compressed_pair_impl(second_type x) : second_{std::move(x)} {}

  [[nodiscard]] first_type& first() noexcept { return *this; }
  [[nodiscard]] const first_type& first() const noexcept { return *this; }

  [[nodiscard]] second_type& second() noexcept { return second_; }
  [[nodiscard]] const second_type& second() const noexcept { return second_; }

 private:
  second_type second_;
};

// selector = 2, U is empty
template <typename T, typename U>
class compressed_pair_impl<T, U, 2> : U {
  using base_type = U;

 public:
  using base_type::base_type;
  using first_type  = T;
  using second_type = U;

  compressed_pair_impl() = default;

  compressed_pair_impl(first_type x, second_type y) : base_type{std::move(y)}, first_{std::move(x)}
  {
  }

  compressed_pair_impl(first_type x) : first_{std::move(x)} {}

  [[nodiscard]] first_type& first() noexcept { return first_; }
  [[nodiscard]] const first_type& first() const noexcept { return first_; }

  [[nodiscard]] second_type& second() noexcept { return *this; }
  [[nodiscard]] const second_type& second() const noexcept { return *this; }

 private:
  first_type first_;
};

// selector = 3, T and U are both empty
template <typename T, typename U>
class compressed_pair_impl<T, U, 3> : T, U {
  using first_base_type  = T;
  using second_base_type = U;

 public:
  using first_type  = T;
  using second_type = U;

  using first_type::first_type;
  using second_type::second_type;

  compressed_pair_impl() = default;

  compressed_pair_impl(first_type x, second_type y)
    : first_type{std::move(x)}, second_type{std::move(y)}
  {
  }

  // Casts are needed to disambiguate case where T or U derive from one another, for example
  //
  // struct T { };
  // struct U : T { };
  //
  // In this case both U and T are able to satisfy "conversion" to T
  [[nodiscard]] first_type& first() noexcept { return static_cast<first_type&>(*this); }
  [[nodiscard]] const first_type& first() const noexcept
  {
    return static_cast<const first_type&>(*this);
  }

  [[nodiscard]] second_type& second() noexcept { return static_cast<second_type&>(*this); }
  [[nodiscard]] const second_type& second() const noexcept
  {
    return static_cast<const second_type&>(*this);
  }
};

}  // namespace detail

template <typename T, typename U>
class compressed_pair
  : public detail::compressed_pair_impl<
      T,
      U,
      detail::compressed_pair_selector<std::is_empty_v<T>, std::is_empty_v<U>>::value> {
  using base_type = detail::compressed_pair_impl<
    T,
    U,
    detail::compressed_pair_selector<std::is_empty_v<T>, std::is_empty_v<U>>::value>;

 public:
  using base_type::base_type;
};

// Intel compilers don't implement empty base optimization (yes, you read that right), so these
// tests fail
#if !defined(__INTEL_COMPILER) && !defined(__ICL)

namespace compressed_pair_test {

struct Empty {};

static_assert(std::is_empty_v<Empty>);
static_assert(sizeof(Empty) == 1);

struct Empty2 {};

static_assert(std::is_empty<Empty2>::value);
static_assert(sizeof(Empty2) == 1);

struct NotEmpty {
  uint64_t d{};
};

static_assert(!std::is_empty_v<NotEmpty>);
static_assert(sizeof(NotEmpty) > 1);

struct EmptyMember {
  Empty m{};
  Empty2 m2{};
};

static_assert(!std::is_empty_v<EmptyMember>);
static_assert(sizeof(EmptyMember) > 1);

// empty-empty should only be 1 byte since both are compressed out
static_assert(std::is_empty_v<compressed_pair<Empty, Empty2>>);
static_assert(sizeof(compressed_pair<Empty, Empty2>) == 1);

// flipping template param order changes nothing
static_assert(std::is_empty_v<compressed_pair<Empty2, Empty>>);
static_assert(sizeof(compressed_pair<Empty2, Empty>) == 1);

// empty-not_empty should be less than sum of sizes, since empty is compressed out
static_assert(!std::is_empty_v<compressed_pair<Empty, NotEmpty>>);
static_assert(sizeof(compressed_pair<Empty, NotEmpty>) < (sizeof(Empty) + sizeof(NotEmpty)));

// flipping template param order changes nothing
static_assert(!std::is_empty_v<compressed_pair<NotEmpty, Empty>>);
static_assert(sizeof(compressed_pair<NotEmpty, Empty>) < (sizeof(NotEmpty) + sizeof(Empty)));

// empty_member-not_empty should also be greater than or equal to sum of sizes (g.t. because
// potential padding) because neither is compressed away
static_assert(!std::is_empty_v<compressed_pair<EmptyMember, NotEmpty>>);
static_assert(sizeof(compressed_pair<EmptyMember, NotEmpty>) >=
              (sizeof(EmptyMember) + sizeof(NotEmpty)));

// flipping template param order changes nothing
static_assert(!std::is_empty_v<compressed_pair<NotEmpty, EmptyMember>>);
static_assert(sizeof(compressed_pair<NotEmpty, EmptyMember>) >=
              (sizeof(NotEmpty) + sizeof(EmptyMember)));

}  // namespace compressed_pair_test

#endif

}  // namespace legate
