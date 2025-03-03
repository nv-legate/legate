/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate.h>

#include <legate/experimental/stl/detail/functional.hpp>
#include <legate/experimental/stl/detail/launch_task.hpp>

// Include this last:
#include <legate/experimental/stl/detail/prefix.hpp>

namespace legate::experimental::stl {

////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Applies the given function `fn` with elements of each of the input
 * sequences `ins` as function arguments.
 *
 * This function launches a Legate task that applies the provided function `fn`
 * element-wise to the pack of ranges `ins`.
 *
 * @param fn The function object to apply with each set of elements.
 * @param ins The input sequences to iterate over.
 *
 * @pre @li The number of input sequences must be greater than 0.
 *      @li The input sequences must satisfy the `logical_store_like` concept.
 *      @li The input sequences must have the same shape.
 *      @li The function object `fn` must be callable with the same number of
 *          arguments as the number of input sequences.
 *      @li The function object `fn` must be trivially copyable.
 *
 * @par Examples:
 * @snippet{trimleft} experimental/stl/for_each.cc stl-for-each-zip-elements
 * @snippet{trimleft} experimental/stl/for_each.cc stl-for-each-zip-by-row
 *
 * @ingroup stl-algorithms
 */
template <typename Function, typename... Inputs>                            //
  requires((sizeof...(Inputs) > 0) && (logical_store_like<Inputs> && ...))  //
void for_each_zip(Function&& fn, Inputs&&... ins)
{
  auto drop_inputs = [fn] LEGATE_HOST_DEVICE(const auto&,  //
                                             const auto&,
                                             auto&& out1,
                                             auto&& out2) {
    return fn(static_cast<decltype(out1)>(out1),  //
              static_cast<decltype(out2)>(out2));
  };
  stl::launch_task(stl::function(drop_inputs),
                   stl::inputs(ins...),
                   stl::outputs(ins...),
                   stl::constraints(stl::align(stl::inputs)));
}

/**
 * @brief Applies the given function to each element in the input range.
 *
 * This function launches a Legate task that applies the provided function `fn` to each element in
 * the input range `input`.
 *
 * @param input The input range to iterate over.
 * @param fn The function to apply to each element.
 *
 * @pre @li The input range `input` must satisfy the `logical_store_like` concept.
 *      @li The function object `fn` must be callable with the element type of the `input` range.
 *      @li The function object `fn` must be trivially copyable.
 *
 * @par Examples:
 * @snippet{trimleft} experimental/stl/for_each.cc stl-for-each-elements
 * @snippet{trimleft} experimental/stl/for_each.cc stl-for-each-by-row
 *
 * @ingroup stl-algorithms
 */
template <typename Input, typename Function>  //
  requires(logical_store_like<Input>)         //
void for_each(Input&& input, Function&& fn)
{
  stl::launch_task(stl::function(drop_n_fn<1>(std::forward<Function>(fn))),
                   stl::inputs(input),
                   stl::outputs(input),
                   stl::constraints(stl::align(stl::inputs)));
}

}  // namespace legate::experimental::stl

#include <legate/experimental/stl/detail/suffix.hpp>
