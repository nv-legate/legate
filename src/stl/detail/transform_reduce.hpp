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

#include "functional.hpp"
#include "prefix.hpp"
#include "reduce.hpp"
#include "stlfwd.hpp"
#include "transform.hpp"

namespace legate::stl {

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename CvrefInput,
          typename Init,
          typename BinaryReduction,
          typename UnaryTransform>                                                  //
  requires(                                                                         //
    logical_store_like<CvrefInput>                                                  //
    && logical_store_like<Init>                                                     //
    && legate_reduction<as_reduction_t<BinaryReduction, element_type_of_t<Init>>>)  //
auto transform_reduce(CvrefInput&& input,
                      Init&& init,
                      BinaryReduction red,
                      UnaryTransform transform)
  -> logical_store<value_type_of_t<Init>, dim_of_v<Init>>
{
  // Check that the operations are trivially relocatable
  detail::check_function_type<BinaryReduction>();
  detail::check_function_type<UnaryTransform>();

  // promote the initial value to the same shape as the input so they can
  // be aligned
  using Reference       = range_reference_t<as_range_t<CvrefInput>>;
  using InputPolicy     = typename std::remove_reference_t<CvrefInput>::policy;
  using TransformResult = value_type_of_t<call_result_t<UnaryTransform, Reference>>;

  as_range_t<CvrefInput> input_rng = as_range(std::forward<CvrefInput>(input));

  auto result = stl::slice_as<InputPolicy>(
    stl::create_store<TransformResult, dim_of_v<CvrefInput>>(input_rng.base().extents()));

  stl::transform(std::forward<as_range_t<CvrefInput>>(input_rng), result, std::move(transform));

  return stl::reduce(result, std::forward<Init>(init), std::move(red));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename CvrefInput1,
          typename CvrefInput2,
          typename Init,
          typename BinaryReduction,
          typename BinaryTransform>                                                        //
  requires(logical_store_like<CvrefInput1>                                                 //
           && logical_store_like<CvrefInput2>                                              //
           && logical_store_like<Init>                                                     //
           && legate_reduction<as_reduction_t<BinaryReduction, element_type_of_t<Init>>>)  //
auto transform_reduce(CvrefInput1&& input1,
                      CvrefInput2&& input2,
                      Init&& init,
                      BinaryReduction red,
                      BinaryTransform transform)
  -> logical_store<element_type_of_t<Init>, dim_of_v<Init>>
{
  // Check that the operations are trivially relocatable
  detail::check_function_type<BinaryReduction>();
  detail::check_function_type<BinaryTransform>();

  static_assert(dim_of_v<CvrefInput1> == dim_of_v<CvrefInput2>);
  static_assert(dim_of_v<CvrefInput1> == dim_of_v<Init> + 1);

  // promote the initial value to the same shape as the input so they can
  // be aligned

  using Reference1      = range_reference_t<as_range_t<CvrefInput1>>;
  using Reference2      = range_reference_t<as_range_t<CvrefInput2>>;
  using InputPolicy     = typename std::remove_reference_t<CvrefInput1>::policy;
  using TransformResult = value_type_of_t<call_result_t<BinaryTransform, Reference1, Reference2>>;

  as_range_t<CvrefInput1> input_rng1 = as_range(std::forward<CvrefInput1>(input1));
  as_range_t<CvrefInput2> input_rng2 = as_range(std::forward<CvrefInput2>(input2));

  LegateAssert(input_rng1.extents() == input_rng2.extents(),
               "Input ranges must have the same extents");

  auto result = stl::slice_as<InputPolicy>(
    stl::create_store<TransformResult, dim_of_v<CvrefInput1>>(input_rng1.base().extents()));

  stl::transform(std::forward<as_range_t<CvrefInput1>>(input_rng1),
                 std::forward<as_range_t<CvrefInput2>>(input_rng2),
                 result,
                 std::move(transform));

  return stl::reduce(result, std::forward<Init>(init), std::move(red));
}

}  // namespace legate::stl

#include "suffix.hpp"
