/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/experimental/stl/detail/functional.hpp>
#include <legate/experimental/stl/detail/reduce.hpp>
#include <legate/experimental/stl/detail/stlfwd.hpp>
#include <legate/experimental/stl/detail/transform.hpp>
#include <legate/utilities/assert.h>

// Include this last:
#include <legate/experimental/stl/detail/prefix.hpp>

namespace legate::experimental::stl {

////////////////////////////////////////////////////////////////////////////////////////////////////
// clang-format off
/**
 * @brief Transform the elements of a store with a unary operation and reduce
 *          them using a binary operation with an initial value.
 *
 * `stl::transform_reduce(input, init, reduction_op, transform_op)` is semantically
 * equivalent (with the caveat noted below) to:
 *
 * @code{.cpp}
 * auto result = stl::create_store<TransformResult>( <extents-of-input> );
 * stl::transform(input, result, transform_op);
 * return stl::reduce(result, init, reduction_op);
 * @endcode
 *
 * `TransformResult` is the result type of the unary transform operation. The
 * input store arguments can be @c legate_store instances, or they can be views
 * created with one of the
 * @verbatim embed:rst:inline :ref:`view adaptors <creating-views>` @endverbatim.
 * If the input store is a view, the `result` store used to hold
 * the results of the transformation step will be a view of the same shape as
 * the input store.
 *
 * @param input The input range to transform.
 * @param init The initial value of the reduction.
 * @param reduction_op The reduction operation to apply to the transformed elements
 *          of the input range. `Reduction` can be a type that satisfies the
 *          @c legate_reduction concept or one of the standard functional objects
 *          `std::plus`, `std::minus`, `std::multiplies`, `std::divides`, etc.;
 *          or an elementwise operation created by passing any of the above to
 *          @c stl::elementwise.
 * @param transform_op The unary operation to apply to the elements of the input
 *          prior to the reduction step.
 *
 * @pre @li `InputRange` must satisfy the @c logical_store_like concept.
 *      @li `Init` must satisfy the @c logical_store_like concept.
 *      @li The result type of the unary transform must be the same as the value
 *          type of the reduction's initial value.
 *      @li The dimension of the input range must be one greater than the
 *          dimension of the initial value.
 *
 * @return An instance of @c logical_store with the same value type and shape as
 *          `init`.
 *
 * @par Example
 * @snippet{trimleft} experimental/stl/transform_reduce.cc 1D unary transform_reduce
 *
 * @see @li @c elementwise
 *      @li @c legate_reduction
 *      @li @ref reduction "Legate's built-in reduction operations"
 * @ingroup stl-algorithms
 */
// clang-format on
template <typename InputRange,
          typename Init,
          typename Reduction,
          typename UnaryTransform>                                            //
  requires(                                                                   //
    logical_store_like<InputRange>                                            //
    && logical_store_like<Init>                                               //
    && legate_reduction<as_reduction_t<Reduction, element_type_of_t<Init>>>)  //
[[nodiscard]] auto transform_reduce(InputRange&& input,
                                    Init&& init,
                                    Reduction&& reduction_op,
                                    UnaryTransform&& transform_op)
  -> logical_store<value_type_of_t<Init>, dim_of_v<Init>>
{
  // Check that the operations are trivially relocatable
  detail::check_function_type<Reduction>();
  detail::check_function_type<UnaryTransform>();

  // promote the initial value to the same shape as the input so they can
  // be aligned
  using Reference       = range_reference_t<as_range_t<InputRange>>;
  using InputPolicy     = typename std::remove_reference_t<InputRange>::policy;
  using TransformResult = value_type_of_t<call_result_t<UnaryTransform, Reference>>;

  // NOLINTNEXTLINE(misc-const-correctness)
  as_range_t<InputRange> input_rng = as_range(std::forward<InputRange>(input));

  auto result =
    stl::slice_as<InputPolicy>(stl::create_store<TransformResult>(input_rng.base().extents()));

  stl::transform(std::move(input_rng), result, std::forward<UnaryTransform>(transform_op));

  return stl::reduce(
    std::move(result), std::forward<Init>(init), std::forward<Reduction>(reduction_op));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// clang-format off
/**
 * @brief Transform the elements of two stores with a binary operation and
 *          reduce them using a binary operation with an initial value.
 *
 * `stl::transform_reduce(input1, input2, init, reduction_op, transform_op)` is
 * semantically equivalent (with the caveat noted below) to:
 *
 * @code{.cpp}
 * auto result = stl::create_store<TransformResult>( <extents-of-input1> );
 * stl::transform(input1, input2, result, transform_op);
 * return stl::reduce(result, init, reduction_op);
 * @endcode
 *
 * `TransformResult` is the result type of the binary transform operation.
 * The input store arguments can be @c legate_store instances, or they can be
 * views created with one of the
 * @verbatim embed:rst:inline :ref:`view adaptors <creating-views>` @endverbatim.
 * If the input stores are views, the `result` store used to hold the results
 * of the transformation step will be a view of the same shape as the first
 * input store.
 *
 * @param input1 The first input range to transform.
 * @param input2 The second input range to transform.
 * @param init The initial value of the reduction.
 * @param reduction_op The reduction operation to apply to the transformed elements
 *          of the input ranges. `Reduction` can be a type that satisfies the
 *          @c legate_reduction concept or one of the standard functional objects
 *          `std::plus`, `std::minus`, `std::multiplies`, `std::divides`, etc.;
 *          or an elementwise operation created by passing any of the above to
 *          @c stl::elementwise.
 * @param transform_op The binary operation to apply to the elements of the two
 *          input ranges prior to the reduction step.
 *
 * @pre @li `InputRange1` and `InputRange2 must satisfy the
 *          @c logical_store_like concept.
 *      @li `Init` must satisfy the @c logical_store_like concept.
 *      @li The shape of the input ranges must be the same.
 *      @li The result type of the binary transform must be the same as the value
 *          type of the initial reduction value.
 *      @li The dimensionality of the input ranges must be one greater than the
 *          dimension of the reduction initial value.
 *
 * @return An instance of @c logical_store with the same value type and shape as
 *          `init`.
 *
 * @see @li @c elementwise
 *      @li @c legate_reduction
 *      @li @ref reduction "Legate's built-in reduction operations"
 * @ingroup stl-algorithms
 */
// clang-format on
template <typename InputRange1,
          typename InputRange2,
          typename Init,
          typename Reduction,
          typename BinaryTransform>                                                  //
  requires(logical_store_like<InputRange1>                                           //
           && logical_store_like<InputRange2>                                        //
           && logical_store_like<Init>                                               //
           && legate_reduction<as_reduction_t<Reduction, element_type_of_t<Init>>>)  //
[[nodiscard]] auto transform_reduce(InputRange1&& input1,
                                    InputRange2&& input2,
                                    Init&& init,
                                    Reduction&& reduction_op,
                                    BinaryTransform&& transform_op)
  -> logical_store<element_type_of_t<Init>, dim_of_v<Init>>
{
  // Check that the operations are trivially relocatable
  detail::check_function_type<Reduction>();
  detail::check_function_type<BinaryTransform>();

  static_assert(dim_of_v<InputRange1> == dim_of_v<InputRange2>);
  static_assert(dim_of_v<InputRange1> == dim_of_v<Init> + 1);

  // promote the initial value to the same shape as the input so they can
  // be aligned

  using Reference1      = range_reference_t<as_range_t<InputRange1>>;
  using Reference2      = range_reference_t<as_range_t<InputRange2>>;
  using InputPolicy     = typename std::remove_reference_t<InputRange1>::policy;
  using TransformResult = value_type_of_t<call_result_t<BinaryTransform, Reference1, Reference2>>;

  as_range_t<InputRange1> input_rng1 = as_range(std::forward<InputRange1>(input1));
  as_range_t<InputRange2> input_rng2 = as_range(std::forward<InputRange2>(input2));

  LEGATE_ASSERT(input_rng1.extents() == input_rng2.extents());

  auto result =
    stl::slice_as<InputPolicy>(stl::create_store<TransformResult>(input_rng1.base().extents()));

  stl::transform(std::forward<as_range_t<InputRange1>>(input_rng1),
                 std::forward<as_range_t<InputRange2>>(input_rng2),
                 result,
                 std::forward<BinaryTransform>(transform_op));

  return stl::reduce(
    std::move(result), std::forward<Init>(init), std::forward<Reduction>(reduction_op));
}

}  // namespace legate::experimental::stl

#include <legate/experimental/stl/detail/suffix.hpp>
