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

#include "core/utilities/defined.h"

#include "config.hpp"
#include "legate.h"
#include "mdspan.hpp"
#include "meta.hpp"
#include "ranges.hpp"
#include "type_traits.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

// Include this last:
#include "prefix.hpp"

/// @defgroup data Data

namespace legate::stl {
////////////////////////////////////////////////////////////////////////////////////////////////////
namespace tags {
inline namespace obj {
}
}  // namespace tags

using namespace tags::obj;

////////////////////////////////////////////////////////////////////////////////////////////////////
using extents                              = const std::size_t[];
inline constexpr std::int32_t dynamic_dims = -1;

////////////////////////////////////////////////////////////////////////////////////////////////////
template <class ElementType, std::int32_t Dim = dynamic_dims>
struct logical_store;

////////////////////////////////////////////////////////////////////////////////////////////////////
namespace detail {
template <class Store>
struct value_type_of_;

template <class ElementType, class Extents, class Layout, class Accessor>
struct value_type_of_<std::mdspan<ElementType, Extents, Layout, Accessor>> {
  using type = ElementType;
};

template <class Storage>
using has_dim_ = meta::constant<!(std::int32_t{Storage::dim()} < 0)>;

template <class Storage>
inline constexpr bool has_dim_v =
  meta::eval<meta::quote_or<has_dim_, std::false_type>, std::remove_reference_t<Storage>>::value;
}  // namespace detail

////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief An alias for the value type of a `legate::stl::logical_store_like` type. A
 *        store's value type is its element type stripped of any `const` qualification.
 *
 * @tparam Storage A type that satisfies the `legate::stl::logical_store_like` concept.
 */
template <class Storage>
using value_type_of_t =
  LEGATE_STL_IMPLEMENTATION_DEFINED(typename detail::value_type_of_<remove_cvref_t<Storage>>::type);

/**
 * @brief An alias for the element type of a `legate::stl::logical_store_like` type. A
 *        store's element type is `const` qualified if the store is read-only.
 *
 * @tparam Storage A type that satisfies the `legate::stl::logical_store_like` concept.
 * @hideinitializer
 */
template <class Storage>
using element_type_of_t = LEGATE_STL_IMPLEMENTATION_DEFINED(
  const_if_t<std::is_const_v<std::remove_reference_t<Storage>>, value_type_of_t<Storage>>);

/**
 * @brief A constexpr variable constant for the dimensionality of a
 *        `legate::stl::logical_store_like` type.
 *
 * @tparam Storage A type that satisfies the `legate::stl::logical_store_like` concept.
 * @hideinitializer
 */
template <class Storage>
  requires(detail::has_dim_v<Storage>)
inline constexpr std::int32_t dim_of_v = std::remove_reference_t<Storage>::dim();

/** @cond */
template <class Storage>
inline constexpr std::int32_t dim_of_v<Storage&> = dim_of_v<Storage>;

template <class Storage>
inline constexpr std::int32_t dim_of_v<Storage&&> = dim_of_v<Storage>;

template <class Storage>
inline constexpr std::int32_t dim_of_v<const Storage> = dim_of_v<Storage>;

template <class ElementType, std::int32_t Dim>
inline constexpr std::int32_t dim_of_v<logical_store<ElementType, Dim>> = Dim;
/** @endcond */

/*************************************************************************************************
 * @brief Given an untyped `legate::LogicalStore`, return a strongly-typed
 *        `legate::stl::logical_store`.
 *
 * @tparam ElementType The element type of the `LogicalStore`.
 * @tparam Dim The dimensionality of the `LogicalStore`.
 * @param store The `LogicalStore` to convert.
 * @return logical_store<ElementType, Dim>
 * @pre The element type of the `LogicalStore` must be the same as `ElementType`,
 *      and the dimensionality of the `LogicalStore` must be the same as `Dim`.
 */
template <class ElementType, std::int32_t Dim = dynamic_dims>
logical_store<ElementType, Dim> as_typed(const legate::LogicalStore& store);

////////////////////////////////////////////////////////////////////////////////////////////////////
namespace detail {
template <class Function, class... InputSpans>
struct elementwise_accessor;

struct default_accessor;

template <class Op, bool Exclusive>
struct reduction_accessor;

template <class ElementType, std::int32_t Dim, class Accessor /*= default_accessor*/>
struct mdspan_accessor;
}  // namespace detail

template <class Input>
using mdspan_for_t = mdspan_t<element_type_of_t<Input>, dim_of_v<Input>>;

/** @cond */
void as_mdspan(const PhysicalStore&&) = delete;
/** @endcond */

/*************************************************************************************************
 * @brief Given an untyped `legate::PhysicalStore`, return a strongly-typed
 *        `legate::stl::logical_store`.
 *
 * @tparam ElementType The element type of the `PhysicalStore`.
 * @tparam Dim The dimensionality of the `PhysicalStore`.
 * @param store The `PhysicalStore` to convert.
 * @return mdspan_t<ElementType, Dim>
 * @pre The element type of the `PhysicalStore` must be the same as
 *      `ElementType`, and the dimensionality of the `Store` must be the same
 *      as `Dim`.
 */
template <class ElementType, std::int32_t Dim>
LEGATE_STL_ATTRIBUTE((host, device))  //
mdspan_t<ElementType, Dim> as_mdspan(const legate::PhysicalStore& store);

/** @overload */
template <class ElementType, std::int32_t Dim>
LEGATE_STL_ATTRIBUTE((host, device))  //
mdspan_t<ElementType, Dim> as_mdspan(const legate::LogicalStore& store);

template <class ElementType, std::int32_t Dim, template <class, std::int32_t> class StoreT>
  requires(same_as<logical_store<ElementType, Dim>, StoreT<ElementType, Dim>>)
LEGATE_STL_ATTRIBUTE((host, device))  //
  mdspan_t<ElementType, Dim> as_mdspan(const StoreT<ElementType, Dim>& store);
/** @endcond */

/** @overload */
template <class ElementType, std::int32_t Dim>
LEGATE_STL_ATTRIBUTE((host, device))  //
mdspan_t<ElementType, Dim> as_mdspan(const legate::PhysicalArray& array);

namespace detail {
struct iteration_kind {};

struct reduction_kind {};

template <class... Types>
void ignore_all(Types&&...);

////////////////////////////////////////////////////////////////////////////////////////////////
template <class StoreLike>
auto logical_store_like_concept_impl(StoreLike& storeish,
                                     LogicalStore& lstore,
                                     mdspan_for_t<StoreLike> span,
                                     PhysicalStore& pstore)  //
  -> decltype(detail::ignore_all(                            //
    StoreLike::policy::logical_view(lstore),
    StoreLike::policy::physical_view(span),
    StoreLike::policy::size(pstore),
    StoreLike::policy::partition_constraints(iteration_kind()),
    StoreLike::policy::partition_constraints(reduction_kind()),
    get_logical_store(storeish)))
{
}

template <class StoreLike, class Ptr = decltype(&logical_store_like_concept_impl<StoreLike>)>
constexpr bool is_logical_store_like(int)
{
  return true;
}

template <class StoreLike>
constexpr bool is_logical_store_like(long)
{
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////
template <class Reduction>
auto legate_reduction_concept_impl(Reduction reduction,
                                   typename Reduction::LHS lhs,  //
                                   typename Reduction::RHS rhs)  //
  -> decltype(detail::ignore_all((                               //
    Reduction::template apply<true>(lhs, std::move(rhs)),        //
    Reduction::template apply<false>(lhs, std::move(rhs)),       //
    Reduction::template fold<true>(rhs, std::move(rhs)),         //
    Reduction::template fold<false>(rhs, std::move(rhs)),        //
    std::integral_constant<typename Reduction::LHS, Reduction::identity>{},
    std::integral_constant<int, Reduction::REDOP_ID>{})))
{
}

template <class Reduction, class Ptr = decltype(&legate_reduction_concept_impl<Reduction>)>
constexpr bool is_legate_reduction(int)
{
  return true;
}

template <class StoreLike>
constexpr bool is_legate_reduction(long)
{
  return false;
}

}  // namespace detail

#if LegateDefined(LEGATE_STL_DOXYGEN)
/**
 * @brief True when `StoreLike` is a type that implements the `get_logical_store`
 *        customization point.
 */
template <class StoreLike>
concept logical_store_like = /* see below */;

/**
 * @brief True when `Reduction` has static `fold` and `apply` member functions.
 */
template <class Reduction>
concept legate_reduction = /* see below */;
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
template <class StoreLike>
inline constexpr bool logical_store_like =
  detail::is_logical_store_like<remove_cvref_t<StoreLike>>(0);

////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Reduction>
inline constexpr bool legate_reduction = detail::is_legate_reduction<remove_cvref_t<Reduction>>(0);

}  // namespace legate::stl

#include "suffix.hpp"
