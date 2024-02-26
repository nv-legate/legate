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

#include "launch_task.hpp"
#include "store.hpp"

// Include this last:
#include "prefix.hpp"

namespace legate::stl {

namespace detail {
template <typename UnaryOperation>
class unary_transform {
 public:
  UnaryOperation op;

  template <typename Src, typename Dst>
  LEGATE_STL_ATTRIBUTE((host, device))
  void operator()(Src&& src, Dst&& dst)
  {
    static_cast<Dst&&>(dst) = op(static_cast<Src&&>(src));
  }
};
template <typename UnaryOperation>
unary_transform(UnaryOperation) -> unary_transform<UnaryOperation>;

template <typename BinaryOperation>
class binary_transform {
 public:
  BinaryOperation op;

  template <typename Src1, typename Src2, typename Dst>
  LEGATE_STL_ATTRIBUTE((host, device))
  void operator()(Src1&& src1, Src2&& src2, Dst&& dst)
  {
    static_assert(std::is_lvalue_reference_v<Dst>);
    static_cast<Dst&&>(dst) = op(static_cast<Src1&&>(src1), static_cast<Src2&&>(src2));
  }
};
template <typename BinaryOperation>
binary_transform(BinaryOperation) -> binary_transform<BinaryOperation>;
}  // namespace detail

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename InputRange,
          typename OutputRange,
          typename UnaryOperation>                                             //
  requires(logical_store_like<InputRange> && logical_store_like<OutputRange>)  //
void transform(InputRange&& input, OutputRange&& output, UnaryOperation op)
{
  detail::check_function_type<UnaryOperation>();
  stl::launch_task(stl::function(detail::unary_transform{std::move(op)}),
                   stl::inputs(input),
                   stl::outputs(output),
                   stl::constraints(stl::align(stl::inputs[0], stl::outputs[0])));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
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

  LegateAssert(input1.extents() == input2.extents());
  LegateAssert(input1.extents() == output.extents());

  stl::launch_task(stl::function(detail::binary_transform{std::move(op)}),
                   stl::inputs(input1, input2),
                   stl::outputs(output),
                   stl::constraints(stl::align(stl::inputs[0], stl::outputs[0]),  //
                                    stl::align(stl::inputs[1], stl::outputs[0])));
}

}  // namespace legate::stl

#include "suffix.hpp"
