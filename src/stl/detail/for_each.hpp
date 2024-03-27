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

#include "functional.hpp"
#include "launch_task.hpp"
#include "legate.h"

// Include this last:
#include "prefix.hpp"

namespace legate::stl {

////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Applies the given function `fn` with elements of each of the input sequences `ins` as
 * function arguments.
 *
 * This function launches a Legate task that applies the provided function `fn` to each element in
 * the input range `input`.
 *
 * @param fn The function object to apply with each set of elements.
 * @param ins The input sequences to iterate over.
 *
 * @requires The number of input sequences must be greater than 0.
 *           The input sequences must satisfy the `logical_store_like` concept.
 */
template <typename Function, typename... Inputs>                            //
  requires((sizeof...(Inputs) > 0) && (logical_store_like<Inputs> && ...))  //
void for_each_zip(Function&& fn, Inputs&&... ins)
{
  stl::launch_task(stl::function(drop_n_fn<sizeof...(Inputs)>(std::forward<Function>(fn))),
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
 * @requires The input range `input` must satisfy the `logical_store_like` concept.
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

}  // namespace legate::stl

#include "suffix.hpp"
