/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// This must go exactly here, because it must come before any Legion/Realm includes. If any of
// those headers come first, then we get extremely confusing errors:
//
// error: implicit instantiation of undefined template 'std::span<const unsigned long, 0>'
//   explicit logical_store(std::span<const std::size_t, 0>) : LogicalStore(logical_store::create())
//   {}
//                                                         ^
// legate/build/debug-sanitizer-clang/_deps/span-src/include/tcb/span.hpp:148:7:
// note: template is declared here
// class span;
//       ^
//
// This type *is* complete and defined at that point! However, Realm has its own span
// implementation, and for whatever reason, this is picked up by the compiler, and used
// instead. You can verify this by compiling the following program:
//
// #include <realm/utils.h>
// #include <realm/instance.h>
// #include <tcb/span.hpp>
//
// int main()
// {
//   std::span<const std::size_t, 0> x;
// }
//
// And you will find the familiar:
//
// span_bug.cpp:11:35: error: implicit instantiation of undefined template
// 'std::span<const unsigned long, 0>'
//   std::span<const std::size_t, 0> x;
//                                   ^
// legate/build/debug-sanitizer-clang/_deps/span-src/include/tcb/span.hpp:148:7:
// note: template is declared here class span;
//       ^
#include <legate/experimental/stl/detail/span.hpp>
//

#include <legate_defines.h>

#include <legate.h>

#include <legate/experimental/stl/detail/config.hpp>
#include <legate/utilities/macros.h>

// As of 3/14/2024, this include causes shadow warnings in GPU debug mode compilation
LEGATE_PRAGMA_PUSH();
LEGATE_PRAGMA_GCC_IGNORE("-Wmaybe-uninitialized");
#include <legate/experimental/stl/detail/mdspan.hpp>
LEGATE_PRAGMA_POP();

#include <legate/experimental/stl/detail/meta.hpp>
#include <legate/experimental/stl/detail/ranges.hpp>
#include <legate/experimental/stl/detail/type_traits.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

// Include this last:
#include <legate/experimental/stl/detail/prefix.hpp>

namespace legate::experimental::stl {
/**
 * @cond
 */

////////////////////////////////////////////////////////////////////////////////////////////////////
namespace tags {

inline namespace obj {
}

}  // namespace tags

// Fully qualify the namespace to ensure that the compiler doesn't pick some other random one
// NOLINTNEXTLINE(google-build-using-namespace)
using namespace ::legate::experimental::stl::tags::obj;

/**
 * @endcond
 */

////////////////////////////////////////////////////////////////////////////////////////////////////
using extents                              = const std::size_t[];
inline constexpr std::int32_t dynamic_dims = -1;  // NOLINT(readability-identifier-naming)

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename ElementType, std::int32_t Dim = dynamic_dims>
class logical_store;  // NOLINT(readability-identifier-naming)

////////////////////////////////////////////////////////////////////////////////////////////////////
namespace detail {

template <typename ElementType, typename Extents, typename Layout, typename Accessor>
struct ValueTypeOf<std::mdspan<ElementType, Extents, Layout, Accessor>> {
  using type = ElementType;
};

template <typename Storage>
using has_dim_ = meta::constant<!(std::int32_t{Storage::dim()} < 0)>;

template <typename Storage>
inline constexpr bool has_dim_v =
  meta::eval<meta::quote_or<has_dim_, std::false_type>, std::remove_reference_t<Storage>>::value;

}  // namespace detail

////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief An alias for the value type of a `legate::experimental::stl::logical_store_like` type. A
 *        store's value type is its element type stripped of any `const` qualification.
 *
 * @tparam Storage A type that satisfies the `legate::experimental::stl::logical_store_like`
 * concept.
 *
 * @hideinitializer
 *
 * @ingroup stl-containers
 */
template <typename Storage>
using value_type_of_t =
  LEGATE_STL_UNSPECIFIED(typename detail::ValueTypeOf<remove_cvref_t<Storage>>::type);

/**
 * @brief An alias for the element type of a `legate::experimental::stl::logical_store_like` type. A
 *        store's element type is `const` qualified if the store is read-only.
 *
 * @tparam Storage A type that satisfies the `legate::experimental::stl::logical_store_like`
 * concept.
 *
 * @hideinitializer
 *
 * @ingroup stl-containers
 */
template <typename Storage>
using element_type_of_t = LEGATE_STL_UNSPECIFIED(
  const_if_t<std::is_const_v<std::remove_reference_t<Storage>>, value_type_of_t<Storage>>);

/**
 * @brief A constexpr variable constant for the dimensionality of a
 *        `legate::experimental::stl::logical_store_like` type.
 *
 * @tparam Storage A type that satisfies the `legate::experimental::stl::logical_store_like`
 * concept.
 *
 * @hideinitializer
 *
 * @ingroup stl-containers
 */
template <typename Storage>
  requires(detail::has_dim_v<Storage>)
inline constexpr std::int32_t dim_of_v = std::remove_reference_t<Storage>::dim();

/** @cond */
template <typename Storage>
inline constexpr std::int32_t dim_of_v<Storage&> = dim_of_v<Storage>;

template <typename Storage>
inline constexpr std::int32_t dim_of_v<Storage&&> = dim_of_v<Storage>;

template <typename Storage>
inline constexpr std::int32_t dim_of_v<const Storage> = dim_of_v<Storage>;

template <typename ElementType, std::int32_t Dim>
inline constexpr std::int32_t dim_of_v<logical_store<ElementType, Dim>> = Dim;
/** @endcond */

////////////////////////////////////////////////////////////////////////////////////////////////////

/** @cond */
template <typename ElementType, std::int32_t Dim = dynamic_dims>
logical_store<ElementType, Dim> as_typed(const legate::LogicalStore& store);
/** @endcond */

namespace detail {

template <typename Function, typename... InputSpans>
class ElementwiseAccessor;

class DefaultAccessor;

template <typename Op, bool Exclusive>
class ReductionAccessor;

template <typename ElementType, std::int32_t Dim, typename Accessor /*= default_accessor*/>
class MDSpanAccessor;

}  // namespace detail

/** @cond */
template <typename Input>
using mdspan_for_t = mdspan_t<element_type_of_t<Input>, dim_of_v<Input>>;
/** @endcond */

/** @cond */
template <typename ElementType, std::int32_t Dim>
LEGATE_HOST_DEVICE [[nodiscard]] mdspan_t<ElementType, Dim> as_mdspan(
  const legate::PhysicalStore& store);

template <typename ElementType, std::int32_t Dim>
LEGATE_HOST_DEVICE [[nodiscard]] mdspan_t<ElementType, Dim> as_mdspan(
  const legate::LogicalStore& store);

template <typename ElementType, std::int32_t Dim, template <typename, std::int32_t> typename StoreT>
  requires(std::is_same_v<logical_store<ElementType, Dim>, StoreT<ElementType, Dim>>)
LEGATE_HOST_DEVICE [[nodiscard]] mdspan_t<ElementType, Dim> as_mdspan(
  const StoreT<ElementType, Dim>& store);

template <typename ElementType, std::int32_t Dim>
LEGATE_HOST_DEVICE [[nodiscard]] mdspan_t<ElementType, Dim> as_mdspan(
  const legate::PhysicalArray& array);

void as_mdspan(const PhysicalStore&&) = delete;
/** @endcond */

struct iteration_kind {};  // NOLINT(readability-identifier-naming)

struct reduction_kind {};  // NOLINT(readability-identifier-naming)

namespace detail {

template <typename... Types>
void ignore_all(Types&&...);

////////////////////////////////////////////////////////////////////////////////////////////////
template <typename StoreLike>
auto logical_store_like_concept_impl(StoreLike& storeish,
                                     LogicalStore& lstore,
                                     mdspan_for_t<StoreLike> span,
                                     PhysicalStore& pstore)  //
  -> decltype(detail::ignore_all(                            //
    StoreLike::policy::logical_view(lstore),
    StoreLike::policy::physical_view(span),
    StoreLike::policy::size(pstore),
    StoreLike::policy::partition_constraints(iteration_kind{}),
    StoreLike::policy::partition_constraints(reduction_kind{}),
    get_logical_store(storeish)))
{
}

template <typename StoreLike, typename Ptr = decltype(&logical_store_like_concept_impl<StoreLike>)>
constexpr bool is_logical_store_like(int)
{
  return true;
}

template <typename StoreLike>
constexpr bool is_logical_store_like(std::int64_t)
{
  return false;
}

static_assert(!std::is_same_v<int, std::int64_t>);

////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Reduction>
auto legate_reduction_concept_impl(Reduction,
                                   typename Reduction::LHS lhs,  //
                                   typename Reduction::RHS rhs)  //
  -> decltype(detail::ignore_all((                               //
    Reduction::template apply<true>(lhs, std::move(rhs)),        //
    Reduction::template apply<false>(lhs, std::move(rhs)),       //
    Reduction::template fold<true>(rhs, std::move(rhs)),         //
    Reduction::template fold<false>(rhs, std::move(rhs)),        //
    // std::integral_constant<int, Reduction::REDOP_ID>{},
    std::integral_constant<typename Reduction::LHS, Reduction::identity>{})))
{
}

template <typename Reduction, typename Ptr = decltype(&legate_reduction_concept_impl<Reduction>)>
constexpr bool is_legate_reduction(int)
{
  return true;
}

template <typename StoreLike>
constexpr bool is_legate_reduction(std::int64_t)
{
  return false;
}

static_assert(!std::is_same_v<int, std::int64_t>);

}  // namespace detail

#if LEGATE_DEFINED(LEGATE_DOXYGEN)
// clang-format off
/**
 * @concept logical_store_like
 *
 * @brief A type `StoreLike` satisfied `logical_store_like` when it exposes a
 * `legate::LogicalStore` via the `get_logical_store` customization point.
 *
 * @code{.cpp}
 * requires(StoreLike& storeish,
 *          legate::LogicalStore& lstore,
 *          stl::mdspan_for_t<StoreLike> span,
 *          legate::PhysicalStore& pstore) {
 *     { get_logical_store(storeish) } -> std::same_as<LogicalStore>;
 *     { StoreLike::policy::logical_view(lstore) } -> std::ranges::range;
 *     { StoreLike::policy::physical_view(span) } -> std::ranges::range;
 *     { StoreLike::policy::size(pstore) } -> legate::coord_t;
 *     { StoreLike::policy::partition_constraints(iteration_kind{}) } -> tuple-like;
 *     { StoreLike::policy::partition_constraints(reduction_kind{}) } -> tuple-like;
 *   };
 * @endcode
 *
 * @see @c get_logical_store
 * @ingroup stl-concepts
 */
template <typename StoreLike>
concept logical_store_like =
  requires(StoreLike& storeish,
           legate::LogicalStore& lstore,
           stl::mdspan_for_t<StoreLike> span,
           legate::PhysicalStore& pstore) {
      { get_logical_store(storeish) } -> std::same_as<LogicalStore>;
      { StoreLike::policy::logical_view(lstore) } -> std::ranges::range;
      { StoreLike::policy::physical_view(span) } -> std::ranges::range;
      { StoreLike::policy::size(pstore) } -> legate::coord_t;
      { StoreLike::policy::partition_constraints(iteration_kind{}) } -> tuple-like;
      { StoreLike::policy::partition_constraints(reduction_kind{}) } -> tuple-like;
    };

/**
 * @concept legate_reduction
 *
 * @brief A concept describing the requirements of a reduction operation that
 * can be used with the @c reduce and @c transform_reduce algorithms.
 *
 * A reduction is characterized by the following three things:
 * @li An `apply` operation
 * @li A `fold` operation
 * @li An identity value
 *
 * `apply` is used to apply the reduction operation to a pair of values,
 * modifying the first value in-place. `fold` is used to combine two values into
 * an accumulator that can then be passed as the second argument to the `apply`
 * operation. The `fold` operation must be reflexive, transitive, and symmetric.
 * `fold`, like `apply`, modifies the first parameter in-place.
 *
 * The following relations must hold for the three reduction components:
 * @li `apply(apply(x, y), z)` is functionally equivalent to `apply(x, fold(y, z))`.
 * @li `apply(x, identity)` leaves `x` unchanged.
 * @li `fold(x, identity)` leaves `x` unchanged.
 *
 * A type `Reduction` satisfies `legate_reduction` if the `requires` clause
 * below is `true`:
 *
 * @code{.cpp}
 * requires (Reduction red, typename Reduction::LHS& lhs, typename Reduction::RHS& rhs) {
 *   { Reduction::template apply<true>(lhs, std::move(rhs)) } -> std::same_as<void>;
 *   { Reduction::template apply<false>(lhs, std::move(rhs)) } -> std::same_as<void>;
 *   { Reduction::template fold<true>(rhs, std::move(rhs)) } -> std::same_as<void>;
 *   { Reduction::template fold<false>(rhs, std::move(rhs)) } -> std::same_as<void>;
 *   typename std::integral_constant<typename Reduction::LHS, Reduction::identity>;
 *   typename std::integral_constant<int, Reduction::REDOP_ID>;
 * }
 * @endcode
 *
 * @see @li @c reduce
 *      @li @c transform_reduce
 * @ingroup stl-concepts
 */
template <typename Reduction>
concept legate_reduction =
  requires (Reduction red, typename Reduction::LHS& lhs, typename Reduction::RHS& rhs) {
    { Reduction::template apply<true>(lhs, std::move(rhs)) } -> std::same_as<void>;
    { Reduction::template apply<false>(lhs, std::move(rhs)) } -> std::same_as<void>;
    { Reduction::template fold<true>(rhs, std::move(rhs)) } -> std::same_as<void>;
    { Reduction::template fold<false>(rhs, std::move(rhs)) } -> std::same_as<void>;
    typename std::integral_constant<typename Reduction::LHS, Reduction::identity>;
    typename std::integral_constant<int, Reduction::REDOP_ID>;
  };
// clang-format on
#endif

/** @cond */
////////////////////////////////////////////////////////////////////////////////////////////////////
// TODO(ericniebler)
// Make these into is_<question>_v
template <typename StoreLike>
inline constexpr bool logical_store_like =  // NOLINT(readability-identifier-naming)
  detail::is_logical_store_like<remove_cvref_t<StoreLike>>(0);

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Reduction>
inline constexpr bool legate_reduction =  // NOLINT(readability-identifier-naming)
  detail::is_legate_reduction<remove_cvref_t<Reduction>>(0);
/** @endcond */

}  // namespace legate::experimental::stl

#include <legate/experimental/stl/detail/suffix.hpp>
