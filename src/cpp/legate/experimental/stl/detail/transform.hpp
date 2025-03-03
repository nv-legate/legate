/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/experimental/stl/detail/launch_task.hpp>
#include <legate/experimental/stl/detail/store.hpp>
#include <legate/utilities/assert.h>

// Include this last:
#include <legate/experimental/stl/detail/prefix.hpp>

namespace legate::experimental::stl {

namespace detail {

template <typename UnaryOperation>
class UnaryTransform {
 public:
  UnaryOperation op{};

  template <typename Src, typename Dst>
  LEGATE_HOST_DEVICE void operator()(Src&& src, Dst&& dst)
  {
    stl::assign(static_cast<Dst&&>(dst), op(static_cast<Src&&>(src)));
  }
};

template <typename UnaryOperation>
UnaryTransform(UnaryOperation) -> UnaryTransform<UnaryOperation>;

template <typename BinaryOperation>
class BinaryTransform {
 public:
  BinaryOperation op{};

  template <typename Src1, typename Src2, typename Dst>
  LEGATE_HOST_DEVICE void operator()(Src1&& src1, Src2&& src2, Dst&& dst)
  {
    stl::assign(static_cast<Dst&&>(dst), op(static_cast<Src1&&>(src1), static_cast<Src2&&>(src2)));
  }
};

template <typename BinaryOperation>
BinaryTransform(BinaryOperation) -> BinaryTransform<BinaryOperation>;

}  // namespace detail

////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Applies a unary operation to each element in the input range and
 * stores the result in the output range.
 *
 * The input range and the output range may be the same.
 *
 * @param input The input range. Must satisfy the @c logical_store_like concept.
 * @param output The output range. Must satisfy the @c logical_store_like concept.
 * @param op The unary operation to apply.
 *
 * @pre @li The input and output ranges must have the same shape.
 *      @li The unary operation must be trivially relocatable.
 *
 * @par Example:
 * @snippet{trimleft} experimental/stl/transform.cc stl-unary-transform-2d
 *
 * @see @c legate::experimental::stl::elementwise
 *
 * @ingroup stl-algorithms
 */
template <typename InputRange, typename OutputRange, typename UnaryOperation>
  requires(logical_store_like<InputRange> && logical_store_like<OutputRange>)  //
void transform(InputRange&& input, OutputRange&& output, UnaryOperation op)
{
  detail::check_function_type<UnaryOperation>();
  /// [stl-launch-task-doxygen-snippet]
  stl::launch_task(stl::function(detail::UnaryTransform{std::move(op)}),
                   stl::inputs(std::forward<InputRange>(input)),
                   stl::outputs(std::forward<OutputRange>(output)),
                   stl::constraints(stl::align(stl::inputs[0], stl::outputs[0])));
  /// [stl-launch-task-doxygen-snippet]
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// clang-format off
/**
 * @brief Applies a binary operation to each element in two input ranges and
 * stores the result in the output range.
 *
 * The output range may be the same as one of the input ranges.
 *
 * @param input1 The first input range. Must satisfy the @c logical_store_like concept.
 * @param input2 The second input range. Must satisfy the @c logical_store_like concept.
 * @param output The output range. Must satisfy the @c logical_store_like concept.
 * @param op The binary operation to apply.
 *
 * @pre @li The input and output ranges must all have the same shape.
 *      @li The binary operation must be trivially relocatable.
 *
 * @par Example:
 * @snippet{trimleft} experimental/stl/transform.cc stl-binary-transform-2d
 *
 * @ingroup stl-algorithms
 */
// clang-format on
template <typename InputRange1,
          typename InputRange2,
          typename OutputRange,
          typename BinaryOperation>             //
  requires(logical_store_like<InputRange1>      //
           && logical_store_like<InputRange2>   //
           && logical_store_like<OutputRange>)  //
void transform(InputRange1&& input1, InputRange2&& input2, OutputRange&& output, BinaryOperation op)
{
  // Check that the operation is trivially relocatable
  detail::check_function_type<BinaryOperation>();

  LEGATE_ASSERT(get_logical_store(input1).extents() == get_logical_store(input2).extents());
  LEGATE_ASSERT(get_logical_store(input1).extents() == get_logical_store(output).extents());

  stl::launch_task(
    stl::function(detail::BinaryTransform{std::move(op)}),
    stl::inputs(std::forward<InputRange1>(input1), std::forward<InputRange2>(input2)),
    stl::outputs(std::forward<OutputRange>(output)),
    stl::constraints(stl::align(stl::inputs[0], stl::outputs[0]),  //
                     stl::align(stl::inputs[1], stl::outputs[0])));
}

}  // namespace legate::experimental::stl

#include <legate/experimental/stl/detail/suffix.hpp>
