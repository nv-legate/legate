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
template <typename Function, typename... Inputs>                            //
  requires((sizeof...(Inputs) > 0) && (logical_store_like<Inputs> && ...))  //
void for_each_zip(Function&& fn, Inputs&&... ins)
{
  stl::launch_task(stl::function(drop_n_fn<sizeof...(Inputs)>(std::forward<Function>(fn))),
                   stl::inputs(ins...),
                   stl::outputs(ins...),
                   stl::constraints(stl::align(stl::inputs)));
}

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
