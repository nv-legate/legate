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

#include "store.hpp"

// Include this last
#include "prefix.hpp"

namespace legate::stl {

////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * Fills the given range with the specified value.
 *
 * This function fills the elements in the range [begin, end) with the specified value.
 * The range must be a logical store-like object, meaning it supports the necessary
 * operations for storing values. The value to be filled is specified by the `val`
 * parameter.
 *
 * @param output The range to be filled.
 * @param val The value to fill the range with.
 */
template <typename Range>              //
  requires(logical_store_like<Range>)  //
void fill(Range&& output, value_type_of_t<Range> val)
{
  auto store                    = get_logical_store(std::forward<Range>(output));
  observer_ptr<Runtime> runtime = legate::Runtime::get_runtime();
  runtime->issue_fill(std::move(store), Scalar{std::move(val)});
}

}  // namespace legate::stl

#include "suffix.hpp"
